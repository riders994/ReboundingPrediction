"""Per-shot player features.

Replaces ``features()`` and ``boxgen()`` in ``coordinator.py``, and the per-shot
``iterrows`` loop in ``obConstructor`` that dominated the old runtime.

Output is long: one row per (shot, player), ten rows per shot. Use
:func:`to_tensor` to reshape into the ``(n_shots, 10, n_features)`` form the Phase 4
model ladder needs.

Notable changes:

* **Coordinates are folded once per shot** with the attacking basket resolved from
  the ball, via :mod:`rebounding.data.court`. The old code applied ``abs(x - 47)``
  independently to each player, which both flipped court handedness and reflected
  back-court players into the front court.
* **Canonical slot ordering**: offense then defense, each sorted by distance to the
  rim *at release*. This gives every column of the flattened tensor a stable
  meaning ("nearest offensive player", "second-nearest defender"), which is what
  lets a plain logistic regression or random forest consume a whole shot at once
  instead of one player at a time. Ordering uses release-time distance because
  that is all the web app has at prediction time -- ordering on rim-time distance
  would not be reproducible when serving.
* **Team rebounds are kept**, labelled via ``IsTeamRebound`` rather than dropped.
  The old ``features()`` returned ``[]`` whenever no individual rebounder matched.
* **Velocity is included.** The old pipeline extracted two isolated frames and never
  computed motion at all. It is worth 1.9 points of top-1 to the rebounder, which is
  less than that omission was once assumed to cost -- see
  :mod:`rebounding.data.derived` on how little of a player's next second his current
  heading explains. The web app cannot supply one, so
  :data:`SERVED_FEATURES` is the variant without it.
* ``boxgen`` asserts its shape instead of silently producing wrong counts when a
  tracking glitch yields other than five players a side.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rebounding.constants import HOOP
from rebounding.data.court import fold, fold_vector, rim_angle, rim_distance
from rebounding.data.derived import CONTEST_DERIVED, DERIVED_FEATURES, SHOT_DERIVED
from rebounding.data.sportvu import GameTracking

N_PLAYERS = 10
TEAM_SIZE = 5

# Window used to estimate velocity by finite difference. At 25 Hz a single frame
# step is far too noisy to differentiate.
VELOCITY_WINDOW_FRAMES = 5

PLAYER_FEATURES = [
    "pre_x", "pre_y", "pre_dist", "pre_angle", "pre_vx", "pre_vy", "pre_speed",
    "pos_x", "pos_y", "pos_dist", "pos_angle",
    "move_dx", "move_dy", "move_dist", "closed_on_rim",
    "pre_cos_shooter", "pos_cos_shooter",
    "pre_box", "pos_box",
    "is_offense", "is_shooter", "role",
]

# Identity features, true at both moments and usable in either regime below.
_STATIC_FEATURES = ["is_offense", "is_shooter", "role"]

# Everything knowable when the ball leaves the shooter's hand. This is what the web
# app can actually supply, because a user places players once and presses go.
RELEASE_FEATURES = [
    "pre_x", "pre_y", "pre_dist", "pre_angle", "pre_vx", "pre_vy", "pre_speed",
    "pre_cos_shooter", "pre_box", *_STATIC_FEATURES,
]

# Everything knowable once the ball reaches the rim, including how each player moved
# to get there. Training on these measures the ceiling, not a servable model: at
# prediction time these positions do not exist yet and would have to be forecast.
# The 2017 work reported 86% top-1 from this regime while serving predicted inputs,
# which is the train/serve skew the two regimes exist to quantify.
RIM_FEATURES = [
    "pos_x", "pos_y", "pos_dist", "pos_angle", "pos_cos_shooter", "pos_box",
    "move_dx", "move_dy", "move_dist", "closed_on_rim", *_STATIC_FEATURES,
]

# Release-time features that survive when nobody supplies a velocity. The web app
# asks a user to place ten players and press go, and the handoff brief settled that
# it will not also ask for ten direction vectors, so this is the set that can
# actually be served. See :mod:`rebounding.data.derived`.
_VELOCITY_COLUMNS = ("pre_vx", "pre_vy", "pre_speed")

# What the app could serve before any of this: positions only, nothing derived.
# Carried as its own regime so the derived features are measured against the set
# they actually replace rather than against one that uses a velocity.
STATIC_FEATURES = [f for f in RELEASE_FEATURES if f not in _VELOCITY_COLUMNS]

# Note what is *not* here: anything about who the players are. The project predicts
# rebounds from location data alone, so per-player history is out of scope even though
# it works -- a smoothed historical rebound rate is worth +1.2 points of top-1, and it
# is excluded on grounds of the question being asked rather than of performance. See the
# README, "Out of scope on purpose". `role` is the one exception, kept for continuity
# with the 2017 model concept; if it ever goes, retrain without it rather than serving a
# default, which is strictly worse than not having the feature.
SERVED_FEATURES = [*STATIC_FEATURES, *CONTEST_DERIVED, *SHOT_DERIVED]

# Release plus everything derivable from it. Needs the frame to have been through
# :meth:`rebounding.data.derived.ShotPriors.transform` first.
RELEASE_DERIVED_FEATURES = [*RELEASE_FEATURES, *DERIVED_FEATURES]

FEATURE_REGIMES = {
    "release": RELEASE_FEATURES,
    "release+derived": RELEASE_DERIVED_FEATURES,
    "static": STATIC_FEATURES,
    "served": SERVED_FEATURES,
    "rim": RIM_FEATURES,
    "all": [*PLAYER_FEATURES, *DERIVED_FEATURES],
}

# The regimes whose features exist without running the derived transform.
BASE_REGIMES = ("release", "rim")


def boxgen(xy: np.ndarray) -> np.ndarray:
    """Crude box-out counts: how many opponents each player is nearest to.

    A single K-means iteration in spirit, kept from the original for continuity.
    ``xy`` must be ``(10, 2)`` ordered with one team in the first five rows.

    The original reshaped to ``(10, 1)`` and sliced ``[:5]`` / ``[5:]`` with no
    check, so a tracking glitch that produced a six/four split returned wrong counts
    without failing.
    """
    if xy.shape != (N_PLAYERS, 2):
        raise ValueError(f"boxgen expects (10, 2), got {xy.shape}")

    first, second = xy[:TEAM_SIZE], xy[TEAM_SIZE:]
    # distances[i, j] = distance from second-team player i to first-team player j
    distances = np.linalg.norm(second[:, None, :] - first[None, :, :], axis=2)
    nearest_first = np.argmin(distances, axis=1)  # for each second-team player
    nearest_second = np.argmin(distances, axis=0)  # for each first-team player

    first_counts = np.bincount(nearest_first, minlength=TEAM_SIZE)
    second_counts = np.bincount(nearest_second, minlength=TEAM_SIZE)
    return np.concatenate([first_counts, second_counts]).astype(np.float32)


def rim_features(
    pre_xy: np.ndarray, pos_xy: np.ndarray, is_shooter: np.ndarray
) -> dict[str, np.ndarray]:
    """The rim-time half of :data:`RIM_FEATURES`, from two sets of positions.

    Batched over shots: every argument carries a leading shot axis, ``pre_xy`` and
    ``pos_xy`` are ``(n_shots, 10, 2)`` in the folded frame and ``is_shooter`` is
    ``(n_shots, 10)``.

    Split out of :func:`shot_features` so that the **movement model's predictions go
    through the identical arithmetic as the tracking data**. The rim-time features are
    the movement model's whole reason for existing, and computing them one way when
    training on real positions and another way when serving predicted ones is the same
    class of mistake as passing zeros for a velocity. There is now one implementation
    and both callers use it.

    ``pos_box`` assumes the canonical slot order -- first five rows one team, last five
    the other -- which :func:`shot_features` establishes and the movement model
    preserves, since it predicts a displacement per slot.
    """
    pre_xy, pos_xy = np.asarray(pre_xy, float), np.asarray(pos_xy, float)
    if pre_xy.shape != pos_xy.shape or pre_xy.shape[-2:] != (N_PLAYERS, 2):
        raise ValueError(
            f"expected matching (n_shots, {N_PLAYERS}, 2), got {pre_xy.shape} and {pos_xy.shape}"
        )

    pre_dist, pos_dist = rim_distance(pre_xy), rim_distance(pos_xy)
    pos_angle = rim_angle(pos_xy)
    shooter_angle = (pos_angle * is_shooter).sum(axis=-1, keepdims=True)
    move = pos_xy - pre_xy

    return {
        "pos_x": pos_xy[..., 0],
        "pos_y": pos_xy[..., 1],
        "pos_dist": pos_dist,
        "pos_angle": pos_angle,
        "pos_cos_shooter": np.cos(pos_angle - shooter_angle),
        "pos_box": np.stack(
            [boxgen(scene) for scene in pos_xy.reshape(-1, N_PLAYERS, 2)]
        ).reshape(pos_dist.shape),
        "move_dx": move[..., 0],
        "move_dy": move[..., 1],
        "move_dist": np.hypot(move[..., 0], move[..., 1]),
        "closed_on_rim": np.where(pos_dist < pre_dist, 1.0, -1.0),
    }


def _velocity(tracking: GameTracking, index: int, quarter: int) -> np.ndarray:
    """Per-player ``(10, 2)`` velocity in ft/s, by backward difference."""
    moments = tracking.moments
    start = max(0, index - VELOCITY_WINDOW_FRAMES)

    # Never differentiate across a quarter boundary.
    quarters = moments["Quarter"].to_numpy()
    while start < index and quarters[start] != quarter:
        start += 1

    dt = float(moments["GameClock"].iat[start] - moments["GameClock"].iat[index])
    if dt <= 0:
        return np.zeros((N_PLAYERS, 2), dtype=np.float32)

    displacement = tracking.player_xyz[index, :, :2] - tracking.player_xyz[start, :, :2]
    return (displacement / dt).astype(np.float32)


def shot_features(tracking: GameTracking, shot: pd.Series) -> pd.DataFrame | None:
    """Ten rows of features for one paired shot, or ``None`` if it cannot be built."""
    release, rim = int(shot["ReleaseIdx"]), int(shot["RimIdx"])
    basket = str(shot["Basket"])

    player_ids = tracking.player_ids[release]
    if not np.array_equal(np.sort(player_ids), np.sort(tracking.player_ids[rim])):
        # Substitution or tracking dropout between the two frames. The old code
        # left-joined here, so a player present at one frame and absent at the other
        # produced NaN feature values that flowed into the training frame.
        return None

    # Reorder the rim frame to match the release frame's player order.
    rim_order = np.array([int(np.flatnonzero(tracking.player_ids[rim] == pid)[0]) for pid in player_ids])

    pre_xy = fold(tracking.player_xyz[release, :, :2], basket)
    pos_xy = fold(tracking.player_xyz[rim][rim_order][:, :2], basket)
    velocity = fold_vector(_velocity(tracking, release, int(shot["Period"])), basket)

    team_ids = tracking.team_ids[release]
    shoot_team = shot.get("ShootTeamID")
    is_offense = (
        (team_ids == int(shoot_team)).astype(np.float32)
        if pd.notna(shoot_team)
        else np.zeros(N_PLAYERS, np.float32)
    )
    if int(is_offense.sum()) != TEAM_SIZE:
        # Without a clean five/five split the box-out counts are meaningless and
        # the canonical ordering has no defined team blocks.
        return None

    shooter_id = shot.get("ShootPlayerID")
    is_shooter = (
        (player_ids == int(shooter_id)).astype(np.float32)
        if pd.notna(shooter_id)
        else np.zeros(N_PLAYERS, np.float32)
    )
    if is_shooter.sum() != 1:
        return None

    # Canonical slots: offence first, then defence, each nearest-to-rim first at
    # release. Deterministic and reproducible from release-time data alone. The sort
    # is applied to the arrays rather than to the assembled frame so that the rim-time
    # columns can come from :func:`rim_features`, which needs the team blocks already
    # in place -- and which is the same code the movement model's predictions go
    # through at serving time.
    order = np.lexsort((rim_distance(pre_xy), -is_offense))
    pre_xy, pos_xy, velocity = pre_xy[order], pos_xy[order], velocity[order]
    player_ids, team_ids = player_ids[order], team_ids[order]
    is_offense, is_shooter = is_offense[order], is_shooter[order]

    pre_dist, pre_angle = rim_distance(pre_xy), rim_angle(pre_xy)
    shooter_slot = int(np.argmax(is_shooter))
    rim_columns = rim_features(pre_xy[None], pos_xy[None], is_shooter[None])

    frame = pd.DataFrame(
        {
            "pre_x": pre_xy[:, 0], "pre_y": pre_xy[:, 1],
            "pre_dist": pre_dist, "pre_angle": pre_angle,
            "pre_vx": velocity[:, 0], "pre_vy": velocity[:, 1],
            "pre_speed": np.hypot(velocity[:, 0], velocity[:, 1]),
            **{name: values[0] for name, values in rim_columns.items()},
            "pre_cos_shooter": np.cos(pre_angle - pre_angle[shooter_slot]),
            "is_offense": is_offense,
            "is_shooter": is_shooter,
            "role": [tracking.roles.get(str(int(pid)), 3.0) for pid in player_ids],
            "PlayerID": player_ids,
            "TeamID": team_ids,
        }
    )
    frame["pre_box"] = boxgen(pre_xy)

    frame["Slot"] = np.arange(N_PLAYERS)
    frame["ShotID"] = shot["ShotID"]
    frame["GameID"] = shot["GameID"]
    frame["FlightTime"] = shot["FlightTime"]
    frame["IsTeamRebound"] = bool(shot["IsTeamRebound"])
    reb_player = shot.get("RebPlayerID")
    frame["Rebounder"] = (
        (frame["PlayerID"] == int(reb_player)).astype(int) if pd.notna(reb_player) else 0
    )

    # A shot whose credited rebounder is not on the floor in the tracking data is
    # unusable as a training example, whether or not it was a team rebound.
    if not frame["IsTeamRebound"].iat[0] and frame["Rebounder"].sum() != 1:
        return None
    return frame


def build(tracking: GameTracking, paired: pd.DataFrame) -> pd.DataFrame:
    """Feature rows for every paired shot in a game."""
    frames = [f for f in (shot_features(tracking, shot) for _, shot in paired.iterrows()) if f is not None]
    if not frames:
        return pd.DataFrame(columns=[*PLAYER_FEATURES, "ShotID", "GameID", "Slot", "Rebounder"])
    return pd.concat(frames, ignore_index=True)


def to_tensor(
    rows: pd.DataFrame, features: list[str] | None = None
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Long rows to ``(n_shots, 10, n_features)`` plus a ``(n_shots, 10)`` label array.

    The canonical ordering in :func:`shot_features` is what makes the middle axis
    mean the same thing across shots, so a flattened ``10 * n_features`` vector is
    coherent for models that cannot handle sets.
    """
    features = features or PLAYER_FEATURES
    ordered = rows.sort_values(["ShotID", "Slot"], kind="stable")
    n_shots = ordered["ShotID"].nunique()

    values = ordered[features].to_numpy(dtype=np.float32)
    labels = ordered["Rebounder"].to_numpy(dtype=np.int8)
    if len(ordered) != n_shots * N_PLAYERS:
        raise ValueError(f"expected {N_PLAYERS} rows per shot, got {len(ordered)} for {n_shots} shots")

    return values.reshape(n_shots, N_PLAYERS, len(features)), labels.reshape(n_shots, N_PLAYERS), features


def hoop_xy() -> tuple[float, float]:
    return HOOP
