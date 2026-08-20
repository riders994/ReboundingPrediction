"""Ten placed dots to ten probabilities: the entry point the web app calls.

Everything else in this package consumes tracking data. This module consumes what a
*user* can supply -- ten positions on a half court, which team is attacking, and who
shot -- and produces the input array
:class:`~rebounding.models.artifact.ModelArtifact` was fitted on. It exists so the app
never reimplements a feature, which is what §3 of ``docs/webapp-handoff.md`` catalogues
going wrong the first time: the app and the pipeline each had their own hoop, their own
angle convention, their own box-out count, and the model was served features it was not
trained on.

Three things this has to get right, all of them silent when wrong.

**Slot order.** The pipeline sorts offense first, then defense, each nearest-to-rim at
release, and every column of the fitted model means something only under that order.
So the ten players are reordered on the way in -- and the probabilities are put back
into the caller's order on the way out. :class:`ShotPrediction` is indexed the way the
caller's list was indexed, never by slot. Returning slot-order probabilities would
mis-assign every dot on the screen while looking entirely reasonable, which is the same
class of bug as the transposed render in §3.4.

**Velocity.** A user places static dots, so there is no velocity to supply.
:meth:`~rebounding.data.derived.ShotPriors.transform` computes velocity-derived
columns regardless, so this module feeds it zeros -- which is safe *only* because none
of those columns survive into ``SERVED_FEATURES``. That invariant is not obvious and
would break quietly if the served list ever grew one, so :func:`predict` checks it and
refuses rather than passing zeros to a model trained on real velocities. The brief is
explicit that the honest no-velocity model beats a velocity model served zeros.

**Coordinate frame.** Inputs are expected in the *folded* frame the pipeline trains
on: ``x`` from 0 at half court to 47 at the baseline, ``y`` in ``[0, 50]``, rim at
``HOOP``. Pass ``basket=`` instead to hand in full-court coordinates and have them
folded here. Positions far outside that frame are rejected, which catches an app
sending unfolded coordinates -- but only when they land in the far half, so it is a
smoke alarm rather than a proof. Confirm handedness once with
:func:`~rebounding.data.court.describe_side_convention`, as §3.5 asks.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from rebounding.constants import COURT_WIDTH, DEFAULT_POSITION, HALF_COURT_X, POSITION_MAP
from rebounding.data.court import fold, rim_angle, rim_distance
from rebounding.data.derived import VELOCITY_DERIVED
from rebounding.data.features import MOVEMENT_SUPPLIED, N_PLAYERS, TEAM_SIZE, boxgen

# Raw velocity columns plus everything derived from one. None of these can be built
# from static dots, so a model that wants any of them cannot be served here.
UNSERVABLE_FEATURES = ("pre_vx", "pre_vy", "pre_speed", *VELOCITY_DERIVED)

# Bounds on a folded position, set from the corpus rather than from the court diagram.
# Over 422,550 training rows, `pre_x` runs -64.9 to 53.9 and `pre_y` runs -2.7 to 52.6:
# players stand behind the baseline (0.25% of rows) and deep in the backcourt (1.3%),
# and tracking noise puts a few just outside the sidelines. These are that measured
# range plus margin, so the check rejects a coordinate frame rather than a position.
X_BOUNDS = (-70.0, 60.0)
Y_BOUNDS = (-10.0, 60.0)


class PlacementError(ValueError):
    """The ten placed players do not describe a shot the model can score."""


@dataclass(frozen=True)
class Player:
    """One placed dot. ``x``/``y`` are folded half-court feet unless ``basket`` is given."""

    x: float
    y: float
    is_offense: bool
    is_shooter: bool = False
    position: str | float | None = None
    player_id: str | None = None

    @property
    def role(self) -> float:
        """Ordinal listed position, defaulting the way the pipeline defaults.

        Accepts a listed-position string (``"G"``, ``"F-C"``) or the ordinal itself,
        since the pipeline stores the ordinal and a caller replaying pipeline rows has
        no string to give back. Anything unrecognised falls back to
        :data:`~rebounding.constants.DEFAULT_POSITION`, as the pipeline does -- the
        original raised ``KeyError`` here and lost the whole game to a blanket except.

        Leaving this unset costs about 1.4 points of top-1, measured on the test split
        by defaulting every player to 3.0. If the UI can ask for positions, it should.
        """
        if self.position is None:
            return DEFAULT_POSITION
        if isinstance(self.position, bool):
            return DEFAULT_POSITION
        if isinstance(self.position, (int, float)):
            return float(self.position)
        return POSITION_MAP.get(str(self.position).upper().strip(), DEFAULT_POSITION)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> Player:
        """Build from the JSON-shaped dict the front end posts."""
        try:
            return cls(
                x=float(data["x"]),
                y=float(data["y"]),
                is_offense=bool(data["is_offense"]),
                is_shooter=bool(data.get("is_shooter", False)),
                position=data.get("position"),
                player_id=data.get("player_id"),
            )
        except KeyError as exc:
            raise PlacementError(f"player is missing required key {exc.args[0]!r}") from exc


def _coerce(players: Iterable[Player | Mapping[str, Any]]) -> list[Player]:
    return [p if isinstance(p, Player) else Player.from_mapping(p) for p in players]


@dataclass(frozen=True)
class ShotPrediction:
    """Per-player rebound probabilities, **indexed as the caller's list was indexed**.

    ``slots`` records which canonical slot each input player was sorted into, which is
    what makes the reordering auditable rather than something to take on trust.
    """

    probabilities: np.ndarray
    slots: np.ndarray
    features: pd.DataFrame

    def ranked(self) -> list[tuple[int, float]]:
        """``(input index, probability)`` most likely first."""
        order = np.argsort(-self.probabilities, kind="stable")
        return [(int(i), float(self.probabilities[i])) for i in order]

    def most_likely(self) -> int:
        """Index into the caller's list of the player most likely to rebound."""
        return int(np.argmax(self.probabilities))


def _validate(players: Sequence[Player]) -> None:
    if len(players) != N_PLAYERS:
        raise PlacementError(f"expected {N_PLAYERS} players, got {len(players)}")

    n_offense = sum(bool(p.is_offense) for p in players)
    if n_offense != TEAM_SIZE:
        # Without a clean five/five the box-out counts are meaningless and the
        # canonical ordering has no defined team blocks -- the pipeline drops such
        # shots outright rather than featurising them.
        raise PlacementError(f"expected {TEAM_SIZE} offensive players, got {n_offense}")

    n_shooter = sum(bool(p.is_shooter) for p in players)
    if n_shooter != 1:
        raise PlacementError(f"expected exactly 1 shooter, got {n_shooter}")
    if not any(p.is_offense and p.is_shooter for p in players):
        raise PlacementError("the shooter must be on the offensive team")


def _check_bounds(xy: np.ndarray) -> None:
    """Catch an app sending unfolded or mis-scaled coordinates.

    Deliberately loose. Its job is to fail an entire coordinate *convention*, not to
    police where a player may stand -- the model was fitted on real tracking data that
    includes the backcourt and a few feet past the baseline, so a tight box here would
    reject placements the training distribution contains.

    It catches full-court coordinates only when they land in the far half. A
    full-court position in the near half looks exactly like a folded backcourt one and
    always will, so confirm handedness once with
    :func:`~rebounding.data.court.describe_side_convention` rather than relying on this.
    """
    x, y = xy[:, 0], xy[:, 1]
    if (x < X_BOUNDS[0]).any() or (x > X_BOUNDS[1]).any():
        raise PlacementError(
            f"x out of range for folded coordinates (got {x.min():.1f}..{x.max():.1f}, "
            f"expected {X_BOUNDS[0]:g}..{X_BOUNDS[1]:g}, with 0 at half court and "
            f"{HALF_COURT_X:g} at the baseline). Full-court coordinates need basket= "
            "so they can be folded."
        )
    if (y < Y_BOUNDS[0]).any() or (y > Y_BOUNDS[1]).any():
        raise PlacementError(
            f"y out of range (got {y.min():.1f}..{y.max():.1f}, expected "
            f"{Y_BOUNDS[0]:g}..{Y_BOUNDS[1]:g} across a {COURT_WIDTH:g} ft floor)"
        )


def feature_frame(
    players: Iterable[Player | Mapping[str, Any]],
    basket: str | None = None,
    shot_id: str = "served",
) -> pd.DataFrame:
    """Ten placed players to the long, canonically ordered frame the pipeline emits.

    The result carries the same base columns ``features.shot_features`` produces, in
    the same slot order, plus ``InputIndex`` recording where each row came from. It
    stops short of the derived columns, which need the artifact's fitted priors.
    """
    placed = _coerce(players)
    _validate(placed)

    xy = np.array([[p.x, p.y] for p in placed], dtype=float)
    if basket is not None:
        xy = fold(xy, basket)
    _check_bounds(xy)

    is_offense = np.array([float(p.is_offense) for p in placed])
    is_shooter = np.array([float(p.is_shooter) for p in placed])
    frame = pd.DataFrame(
        {
            "InputIndex": np.arange(len(placed)),
            "pre_x": xy[:, 0],
            "pre_y": xy[:, 1],
            "pre_dist": rim_distance(xy),
            "pre_angle": rim_angle(xy),
            "is_offense": is_offense,
            "is_shooter": is_shooter,
            "role": [p.role for p in placed],
            "PlayerID": [p.player_id for p in placed],
        }
    )

    # Canonical slots, exactly as features.shot_features orders them: offense first,
    # then defense, each nearest-to-rim at release.
    frame = frame.sort_values(
        ["is_offense", "pre_dist"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)

    shooter_angle = float(frame.loc[frame["is_shooter"] == 1.0, "pre_angle"].iloc[0])
    frame["pre_cos_shooter"] = np.cos(frame["pre_angle"] - shooter_angle)
    # After the sort, so the two team blocks are where boxgen expects them.
    frame["pre_box"] = boxgen(frame[["pre_x", "pre_y"]].to_numpy())

    # A user places static dots. These are zeros so ShotPriors.transform can run;
    # predict() is what guarantees nothing built from them reaches the model.
    frame["pre_vx"] = 0.0
    frame["pre_vy"] = 0.0
    frame["pre_speed"] = 0.0

    frame["Slot"] = np.arange(len(frame))
    frame["ShotID"] = shot_id
    return frame


@dataclass(frozen=True)
class MovementPrediction:
    """Where the ten players go while the ball is in the air, **in caller order**.

    ``scenes`` is ``(n, 10, 2)`` of sampled futures and is what the app should draw --
    one per animation, or several at once as ghosts to show the spread. ``mean`` is
    the conditional-mean scene, kept for the rebounder and for tests; drawing it is
    the 2017 mistake, because the average of crashing the glass and leaking out is a
    player standing in neither place.
    """

    scenes: np.ndarray
    mean: np.ndarray
    slots: np.ndarray
    features: pd.DataFrame

    def scene(self, index: int = 0) -> np.ndarray:
        """One sampled future, ``(10, 2)``, indexed as the caller's list was."""
        return self.scenes[index]


def _served_tensor(artifact, frame: pd.DataFrame, wanted: Sequence[str]) -> np.ndarray:
    missing = [name for name in wanted if name not in frame.columns]
    if missing:
        raise PlacementError(f"could not build {missing} for this artifact")
    return frame[list(wanted)].to_numpy(dtype=np.float32).reshape(1, N_PLAYERS, len(wanted))


def animate(
    movement,
    players: Iterable[Player | Mapping[str, Any]],
    basket: str | None = None,
    n: int = 20,
    seed: int | None = None,
) -> MovementPrediction:
    """Run the movement model on ten placed players.

    ``movement`` is a :class:`~rebounding.models.artifact.MovementArtifact`. This is
    the app's other entry point beside :func:`predict`, and it replaces ``posnn.h5``
    and the eight hand-built columns the 2017 ``features()`` fed it.

    Sampling ``n`` scenes and drawing them is the intended use. A generative model
    that gets averaged back down to one dot before it reaches the screen has had its
    only advantage discarded on the last step.
    """
    frame = feature_frame(players, basket=basket)
    enriched = movement.priors.transform(frame)
    x = _served_tensor(movement, enriched, movement.features)
    release = enriched[["pre_x", "pre_y"]].to_numpy(dtype=float).reshape(1, N_PLAYERS, 2)

    by_slot_scenes = movement.sample_positions(x, release, n=n, seed=seed)[:, 0]
    by_slot_mean = movement.predict_positions(x, release)[0]

    input_index = enriched["InputIndex"].to_numpy(dtype=int)
    scenes = np.empty_like(by_slot_scenes)
    scenes[:, input_index] = by_slot_scenes
    mean = np.empty_like(by_slot_mean)
    mean[input_index] = by_slot_mean

    slots = np.empty(len(input_index), dtype=int)
    slots[input_index] = enriched["Slot"].to_numpy(dtype=int)
    return MovementPrediction(scenes=scenes, mean=mean, slots=slots, features=enriched)


def predict(
    artifact,
    players: Iterable[Player | Mapping[str, Any]],
    basket: str | None = None,
    movement=None,
) -> ShotPrediction:
    """Score one placed shot with a loaded artifact.

    ``artifact`` is a :class:`~rebounding.models.artifact.ModelArtifact`. Its own
    ``features`` list drives the column order, not the imported ``SERVED_FEATURES`` --
    the artifact is the record of what the trees were actually fitted on.

    An artifact fitted on the ``served+movement`` regime needs ``movement``: its last
    ten columns are rim-time positions that only the movement model can supply. It is
    refused rather than defaulted, for the same reason a velocity-hungry artifact is.
    """
    wanted = list(artifact.features)
    unservable = [name for name in wanted if name in UNSERVABLE_FEATURES]
    if unservable:
        # Serving zeros here would be train/serve skew, and strictly worse than the
        # honest model fitted without velocity. Refuse instead of degrading quietly.
        raise PlacementError(
            f"this artifact needs {unservable}, which cannot be built from static "
            "positions. Serve a model fitted on SERVED_FEATURES instead of feeding it "
            "zero velocities."
        )

    needs_movement = [name for name in wanted if name in MOVEMENT_SUPPLIED]
    if needs_movement and movement is None:
        raise PlacementError(
            f"this artifact needs {needs_movement}, which are rim-time positions. Pass "
            "a MovementArtifact as `movement=` -- these cannot be read off a placed "
            "shot, because the shot has not landed yet."
        )

    frame = feature_frame(players, basket=basket)
    enriched = artifact.priors.transform(frame)

    # to_tensor is not usable here: it reads a `Rebounder` label column, which does
    # not exist for a shot that has not happened.
    if needs_movement:
        served = _served_tensor(movement, enriched, movement.features)
        release = enriched[["pre_x", "pre_y"]].to_numpy(dtype=float).reshape(1, N_PLAYERS, 2)
        shooter = enriched["is_shooter"].to_numpy(dtype=float).reshape(1, N_PLAYERS)
        block = movement.rim_block(served, release, shooter)
        for position, name in enumerate(MOVEMENT_SUPPLIED):
            enriched[name] = block[0, :, position]

    x = _served_tensor(artifact, enriched, wanted)
    by_slot = artifact.predict_proba(x)[0]

    # Back into the caller's order. This is the line that keeps the returned
    # probabilities attached to the dots the user actually placed.
    input_index = enriched["InputIndex"].to_numpy(dtype=int)
    probabilities = np.empty_like(by_slot)
    probabilities[input_index] = by_slot

    slots = np.empty(len(input_index), dtype=int)
    slots[input_index] = enriched["Slot"].to_numpy(dtype=int)
    return ShotPrediction(probabilities=probabilities, slots=slots, features=enriched)
