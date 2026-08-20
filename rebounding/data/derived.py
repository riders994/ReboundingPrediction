"""Features built from the per-shot frame, not from the tracking corpus.

Everything in :mod:`rebounding.data.features` describes a player on his own: where he
is, how fast he is going, how far he is from the rim. Rebounding is a competition
between ten of them, so the quantities that decide it are relative -- who is inside
whom, who has space, who is nearest the place the ball is going to land. None of that
is visible in a row that only knows about one player.

These are derived rather than extracted, which matters for two reasons. They are a
pure function of the ten positions, the ten velocities, who shot, and which team is
attacking, so **the web app can compute every one of them at serving time** from what
a user places on the court. And they need no rebuild of the 3.6 GB tracking corpus,
so they can be added to an existing ``frame.parquet``.

Measured with :class:`rebounding.models.boosted.BoostedSoftmax`, scored per shot as
top-1 of ten. The README carries the full ladder; the rows that matter here are what
these features are worth on top of the sets they extend:

=================================================  ==========  ==========
feature set                                        validation        test
=================================================  ==========  ==========
positions only (``STATIC_FEATURES``)                    27.1%       27.5%
positions + derived (``SERVED_FEATURES``)               28.8%       29.2%
release, with velocity (``RELEASE_FEATURES``)           28.5%       28.5%
release + derived (``RELEASE_DERIVED_FEATURES``)        29.9%       31.1%
=================================================  ==========  ==========

The second row is the one the web app can serve, and it beats the first without asking
a user for anything new.

Two things deliberately absent.

**Flight time is predicted, not measured.** ``FlightTime`` in the frame is the
tracking-measured release-to-rim interval, which at serving time has not happened
yet. :class:`ShotPriors` predicts it from the shot distance instead. It is only 47%
predictable that way, but substituting the prediction costs 0.23 points and is not
distinguishable from using the measured value, so there is no reason to serve a
feature that cannot exist.

**No spatial target encoding.** A smoothed grid of "how often does a player standing
here get the board", fitted per shot-distance bucket and team, was measured at -0.02
points against the boosted model without it. Gradient boosting over ``pre_x``,
``pre_y``, ``is_offense`` and ``shot_dist`` already recovers whatever the grid
encodes, and the encoding brings leakage risk for nothing. It is recorded here as
tried rather than left as an obvious idea nobody tested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rebounding.constants import HOOP

N_PLAYERS = 10

# Radius over which "how contested is this player" is measured. Roughly the distance
# a player can cover while the ball is in the air.
CROWDING_RADII = (6.0, 10.0)

# Scale of the soft nearest-man weighting, in feet.
CLOSENESS_SCALE = 6.0


def _cube(rows: pd.DataFrame, columns: list[str]) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Long rows to one ``(n_shots, 10)`` array per column, in canonical slot order."""
    ordered = rows.sort_values(["ShotID", "Slot"], kind="stable")
    if len(ordered) % N_PLAYERS:
        raise ValueError(f"expected a multiple of {N_PLAYERS} rows, got {len(ordered)}")
    n_shots = len(ordered) // N_PLAYERS
    return (
        {c: ordered[c].to_numpy(np.float64).reshape(n_shots, N_PLAYERS) for c in columns},
        ordered,
    )


def motion_features(x, y, vx, vy, dist) -> dict[str, np.ndarray]:
    """Velocity resolved against the basket rather than against the court axes.

    ``pre_vx`` and ``pre_vy`` are in court coordinates, so the same pair of numbers
    means "closing on the rim" for a player on one wing and "leaving it" for a player
    on the other. Rotating into the rim's frame makes the sign mean one thing
    everywhere, which is what a tree can split on.
    """
    outward_x = (x - HOOP[0]) / np.maximum(dist, 1e-6)
    outward_y = (y - HOOP[1]) / np.maximum(dist, 1e-6)
    return {
        # positive when the player is closing on the rim
        "v_radial": -(vx * outward_x + vy * outward_y),
        # signed, around the basket: which way he is circling
        "v_tangent": vx * outward_y - vy * outward_x,
    }


def extrapolate(x, y, vx, vy, flight, damping: float) -> dict[str, np.ndarray]:
    """Where each player would be at rim time if he carried on in a straight line.

    ``damping`` is fitted rather than assumed, because full velocity is *worse than
    useless*: over a 1.9 second flight, extrapolating at the measured velocity lands
    8.46 ft from the truth while assuming the player never moves lands 8.41 ft away.
    At the fitted damping of about 0.38 the error falls to 6.49 ft against the 7.67 ft
    a player actually travels. A player's release-time velocity survives roughly four
    tenths of a second of NBA movement; see the README on what that implies for a
    movement model.
    """
    step_x, step_y = damping * vx * flight, damping * vy * flight
    ex, ey = x + step_x, y + step_y
    dist = np.hypot(ex - HOOP[0], ey - HOOP[1])
    return {
        "ext_x": ex,
        "ext_y": ey,
        "ext_dist": dist,
        "ext_closing": np.hypot(x - HOOP[0], y - HOOP[1]) - dist,
        "ext_minus_best": dist - dist.min(axis=1, keepdims=True),
    }


def shot_context(dist, angle, is_shooter, shot_distance) -> dict[str, np.ndarray]:
    """The shot itself, made per-player so a grouped softmax can use it.

    A quantity that is the same for all ten players cancels out of a softmax taken
    over the ten, so shot distance on its own contributes exactly nothing. It has to
    enter through something that varies down the group: a bearing measured relative
    to the shot, or a product with the player's own distance.
    """
    shot_angle = (angle * is_shooter).sum(axis=1, keepdims=True)
    relative = np.arctan2(np.sin(angle - shot_angle), np.cos(angle - shot_angle))
    return {
        "shot_dist": shot_distance,
        # Signed bearing off the shot's line. `pre_cos_shooter` already carries the
        # unsigned part; the sine is what separates the strong side from the weak
        # side, which a cosine cannot see.
        "rel_bearing": relative,
        "abs_rel_bearing": np.abs(relative),
        "sin_shooter": np.sin(relative),
        # Long shots come off long. The product is what lets a linear score express
        # "be further out when the shot is further out".
        "dist_x_shot": dist * shot_distance / 25.0,
    }


def contest_features(x, y, dist, is_offense) -> dict[str, np.ndarray]:
    """Who is inside whom, who has space, and how the ten rank against each other.

    ``pre_box`` already counts how many opponents a player is nearest to, which is a
    crude proxy for boxing out. These are the quantities it stands in for: the actual
    gap to the nearest opponent, whether that opponent is between the player and the
    basket, and how the player ranks against the other nine rather than in absolute
    feet.
    """
    offsets_x = x[:, :, None] - x[:, None, :]
    offsets_y = y[:, :, None] - y[:, None, :]
    separation = np.hypot(offsets_x, offsets_y)  # (n_shots, 10, 10)

    same_team = is_offense[:, :, None] == is_offense[:, None, :]
    is_self = np.eye(N_PLAYERS, dtype=bool)[None, :, :]
    far = 1e6

    opponents = np.where(~same_team, separation, far)
    nearest_opponent = opponents.argmin(axis=2)
    anyone = np.where(is_self, far, separation)

    outward_x = (x - HOOP[0]) / np.maximum(dist, 1e-6)
    outward_y = (y - HOOP[1]) / np.maximum(dist, 1e-6)
    to_opponent_x = np.take_along_axis(x, nearest_opponent, 1) - x
    to_opponent_y = np.take_along_axis(y, nearest_opponent, 1) - y

    teammates = np.where(same_team & ~is_self, dist[:, None, :], far)

    out = {
        "d_nearest_opp": np.take_along_axis(opponents, nearest_opponent[:, :, None], 2)[:, :, 0],
        "d_nearest_any": anyone.min(axis=2),
        # Positive when the player is nearer the basket than the man guarding him.
        "inside_gap": np.take_along_axis(dist, nearest_opponent, 1) - dist,
        # Projection of the nearest opponent's offset onto the line to the basket:
        # positive means he is genuinely between this player and the rim.
        "opp_boxes_me": to_opponent_x * -outward_x + to_opponent_y * -outward_y,
        "n_opp_inside": np.where(~same_team, dist[:, None, :] < dist[:, :, None], False)
        .sum(axis=2)
        .astype(float),
        "n_team_inside": np.where(same_team & ~is_self, dist[:, None, :] < dist[:, :, None], False)
        .sum(axis=2)
        .astype(float),
        # Against the whole floor, this player included, so the nearest man scores 0.
        "dist_minus_best": dist - dist.min(axis=1, keepdims=True),
        # Against his four teammates, himself excluded, so it goes negative for the
        # nearest man of a team and says by how much. Including himself would flatten
        # every team's leader to 0 and throw away exactly that.
        "dist_minus_nearest_teammate": dist - teammates.min(axis=2),
        "dist_minus_mean": dist - dist.mean(axis=1, keepdims=True),
    }
    for radius in CROWDING_RADII:
        out[f"n_within_{int(radius)}"] = (anyone < radius).sum(axis=2).astype(float)

    # A soft version of "is the nearest man": degrades gracefully when two players
    # are level, where the hard argmin flips on a foot of tracking noise.
    weight = np.exp(-dist / CLOSENESS_SCALE)
    out["closeness_share"] = weight / weight.sum(axis=1, keepdims=True)
    return out


class ShotPriors:
    """The two quantities that have to be *fitted* rather than computed.

    Fitted on the training games only and applied to the others, so neither the
    validation nor the test split contributes to its own features.
    """

    def __init__(self) -> None:
        self.damping_: float | None = None
        self.flight_coefficients_: np.ndarray | None = None

    def fit(self, rows: pd.DataFrame) -> ShotPriors:
        d, _ = _cube(
            rows,
            ["pre_x", "pre_y", "pre_vx", "pre_vy", "pos_x", "pos_y", "FlightTime",
             "pre_dist", "is_shooter"],
        )
        step_x = d["pre_vx"] * d["FlightTime"]
        step_y = d["pre_vy"] * d["FlightTime"]
        move_x = d["pos_x"] - d["pre_x"]
        move_y = d["pos_y"] - d["pre_y"]
        self.damping_ = float(
            (move_x * step_x + move_y * step_y).sum() / (step_x**2 + step_y**2).sum()
        )

        shot_distance = (d["pre_dist"] * d["is_shooter"]).sum(axis=1)
        flight = d["FlightTime"][:, 0]
        design = np.column_stack([np.ones_like(shot_distance), shot_distance, shot_distance**2])
        self.flight_coefficients_ = np.linalg.lstsq(design, flight, rcond=None)[0]
        return self

    def predicted_flight(self, shot_distance: np.ndarray) -> np.ndarray:
        """Predicted release-to-rim time, held flat past the distance where it peaks.

        Flight time **saturates**. Binned over the training games it climbs from 1.12 s
        inside 5 ft to 2.33 s by 30 ft and then stops: 2.34 s over 30-35 ft, 2.24 s over
        35-40 ft. Past about 28 ft the shot is taken on a flatter, harder trajectory and
        gains no more hang time.

        A quadratic cannot express a plateau -- it has to turn over -- so the raw fit
        predicts a 48.7 ft shot (the furthest a user can place a shooter on the web app's
        half-court canvas) hanging 1.46 s, less than a ten-footer, and goes negative past
        92 ft. Holding the peak fixes that, and it is not a patch over a bad fit: on the
        validation split the clamped form scores 0.4035 RMSE against the unclamped
        0.4037, and its plateau of 2.28 s lands within 0.05 s of the measured one.

        Raising the degree does not help, which is worth recording so nobody retries it.
        Degrees 2, 3, 4 and 6 all sit between 0.4032 and 0.4037 validation RMSE -- a
        difference of half a millisecond -- and every one of them is non-monotone. The
        higher degrees are worse where it matters: at 62 ft the quartic predicts 5.17 s
        and the sextic 9.46 s. Saturating forms (``a + b*log(1+d)``, ``a + b*sqrt(d)``,
        ``a + b*(1 - exp(-d/k))``) are all monotone but fit no better, and the first two
        keep climbing where the data flattens.
        """
        if self.flight_coefficients_ is None:
            raise RuntimeError("priors are not fitted")
        a, b, c = self.flight_coefficients_
        distance = np.asarray(shot_distance, dtype=float)
        if c < 0:
            # Downward parabola: the vertex is the peak, so clamp the input to it.
            distance = np.minimum(distance, -b / (2.0 * c))
        return a + b * distance + c * distance**2

    def transform(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Return ``rows`` in canonical order with every derived column appended."""
        if self.damping_ is None:
            raise RuntimeError("priors are not fitted")

        d, ordered = _cube(
            rows,
            ["pre_x", "pre_y", "pre_vx", "pre_vy", "pre_dist", "pre_angle",
             "is_offense", "is_shooter"],
        )
        shot_distance = np.repeat(
            (d["pre_dist"] * d["is_shooter"]).sum(axis=1, keepdims=True), N_PLAYERS, axis=1
        )
        flight = self.predicted_flight(shot_distance)

        columns: dict[str, np.ndarray] = {"flight_hat": flight}
        columns.update(
            motion_features(d["pre_x"], d["pre_y"], d["pre_vx"], d["pre_vy"], d["pre_dist"])
        )
        columns.update(
            extrapolate(d["pre_x"], d["pre_y"], d["pre_vx"], d["pre_vy"], flight, self.damping_)
        )
        columns.update(
            shot_context(d["pre_dist"], d["pre_angle"], d["is_shooter"], shot_distance)
        )
        columns.update(contest_features(d["pre_x"], d["pre_y"], d["pre_dist"], d["is_offense"]))

        frame = ordered.copy()
        for name, values in columns.items():
            frame[name] = values.reshape(-1)
        return frame


# Derived features that need a velocity to compute. The web app does not collect one
# -- a user places ten players and presses go -- so the servable feature set is the
# one below with these removed.
VELOCITY_DERIVED = [
    "v_radial", "v_tangent",
    "ext_x", "ext_y", "ext_dist", "ext_closing", "ext_minus_best",
]

CONTEST_DERIVED = [
    "d_nearest_opp", "d_nearest_any", "n_within_6", "n_within_10",
    "inside_gap", "opp_boxes_me", "n_opp_inside", "n_team_inside",
    "dist_minus_best", "dist_minus_nearest_teammate", "dist_minus_mean", "closeness_share",
]

SHOT_DERIVED = [
    "flight_hat", "shot_dist", "rel_bearing", "abs_rel_bearing", "sin_shooter", "dist_x_shot",
]

DERIVED_FEATURES = VELOCITY_DERIVED + CONTEST_DERIVED + SHOT_DERIVED
