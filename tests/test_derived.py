"""Tests for the derived, relative features.

These are geometry, so every one of them has a right answer that can be worked out
on paper. The scenes below are built by hand for that reason: a feature named
``inside_gap`` that comes out negative when the player is demonstrably inside his man
is wrong no matter what it scores on the real frame.
"""

import numpy as np
import pandas as pd
import pytest

from rebounding.constants import HOOP
from rebounding.data.derived import (
    CONTEST_DERIVED,
    ShotPriors,
    contest_features,
    extrapolate,
    motion_features,
    shot_context,
)
from rebounding.data.features import RELEASE_FEATURES, SERVED_FEATURES, STATIC_FEATURES

N_PLAYERS = 10


def scene(positions, is_offense):
    """``(x, y, dist, is_offense)`` arrays for one hand-built shot."""
    xy = np.asarray(positions, dtype=float)[None, :, :]
    x, y = xy[:, :, 0], xy[:, :, 1]
    dist = np.hypot(x - HOOP[0], y - HOOP[1])
    return x, y, dist, np.asarray(is_offense, dtype=float)[None, :]


@pytest.fixture
def frame():
    """A small frame with the columns the transform needs, two shots of ten."""
    rng = np.random.default_rng(0)
    n_shots = 40
    rows = []
    for shot in range(n_shots):
        shooter = rng.integers(0, 5)
        for slot in range(N_PLAYERS):
            x, y = rng.uniform(5, 45), rng.uniform(2, 48)
            rows.append(
                {
                    "ShotID": f"s{shot}",
                    "GameID": f"g{shot % 4}",
                    "Slot": slot,
                    "pre_x": x,
                    "pre_y": y,
                    "pre_vx": rng.normal(0, 3),
                    "pre_vy": rng.normal(0, 3),
                    "pre_dist": np.hypot(x - HOOP[0], y - HOOP[1]),
                    "pre_angle": np.arctan2(y - HOOP[1], -(x - HOOP[0])),
                    "pos_x": x + rng.normal(0, 4),
                    "pos_y": y + rng.normal(0, 4),
                    "FlightTime": 1.0 + 0.05 * shot % 2,
                    "is_offense": float(slot < 5),
                    "is_shooter": float(slot == shooter),
                    "Rebounder": int(slot == (shot % N_PLAYERS)),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Contest geometry
# --------------------------------------------------------------------------- #


def test_inside_gap_is_positive_for_the_player_nearer_the_basket():
    """Slot 0 stands between his man and the rim, so his gap must be positive."""
    positions = [(HOOP[0] - 4, 25), *[(0, 40 + i) for i in range(4)],
                 (HOOP[0] - 10, 25), *[(0, 5 + i) for i in range(4)]]
    x, y, dist, is_offense = scene(positions, [1] * 5 + [0] * 5)
    out = contest_features(x, y, dist, is_offense)

    assert out["inside_gap"][0, 0] == pytest.approx(6.0)
    assert out["inside_gap"][0, 5] == pytest.approx(-6.0)
    # and the defender is genuinely behind him, not merely further out
    assert out["opp_boxes_me"][0, 0] < 0
    assert out["opp_boxes_me"][0, 5] > 0


def test_counts_only_look_at_the_right_team():
    """Four opponents inside him, and none of his own team."""
    inside = [(HOOP[0] - 2 - i, 25) for i in range(4)]
    positions = [(HOOP[0] - 20, 25), *[(2, 40 + i) for i in range(4)], *inside, (2, 5)]
    x, y, dist, is_offense = scene(positions, [1] * 5 + [0] * 5)
    out = contest_features(x, y, dist, is_offense)

    assert out["n_opp_inside"][0, 0] == 4
    assert out["n_team_inside"][0, 0] == 0


def test_nearest_distances_exclude_the_player_himself():
    positions = [(10, 25), (13, 25), *[(40, 5 + i) for i in range(3)],
                 *[(45, 45 - i) for i in range(5)]]
    x, y, dist, is_offense = scene(positions, [1] * 5 + [0] * 5)
    out = contest_features(x, y, dist, is_offense)

    assert out["d_nearest_any"][0, 0] == pytest.approx(3.0)
    assert out["d_nearest_any"][0, 0] > 0  # never zero from matching itself
    # nearest *opponent* is much further away than the nearest teammate
    assert out["d_nearest_opp"][0, 0] > out["d_nearest_any"][0, 0]


def test_relative_distances_are_measured_against_the_right_field():
    positions = [(HOOP[0] - 5, 25), *[(HOOP[0] - 15, 20 + i) for i in range(4)],
                 (HOOP[0] - 2, 25), *[(HOOP[0] - 20, 30 + i) for i in range(4)]]
    x, y, dist, is_offense = scene(positions, [1] * 5 + [0] * 5)
    out = contest_features(x, y, dist, is_offense)

    # the defender at 2 ft is the closest man on the floor
    assert out["dist_minus_best"][0, 5] == pytest.approx(0.0)
    assert out["dist_minus_best"][0, 0] == pytest.approx(3.0)
    # slot 0 is the closest of his own team, and the feature says by how much
    assert out["dist_minus_nearest_teammate"][0, 0] < 0


def test_closeness_share_sums_to_one_over_the_ten():
    rng = np.random.default_rng(3)
    xy = rng.uniform(0, 45, size=(6, N_PLAYERS, 2))
    x, y = xy[:, :, 0], xy[:, :, 1]
    dist = np.hypot(x - HOOP[0], y - HOOP[1])
    is_offense = np.tile([1.0] * 5 + [0.0] * 5, (6, 1))
    out = contest_features(x, y, dist, is_offense)
    assert np.allclose(out["closeness_share"].sum(axis=1), 1.0)


# --------------------------------------------------------------------------- #
# Motion
# --------------------------------------------------------------------------- #


def test_radial_velocity_is_positive_when_closing_on_the_rim():
    x = np.array([[HOOP[0] - 10.0, HOOP[0] - 10.0]])
    y = np.array([[25.0, 25.0]])
    dist = np.array([[10.0, 10.0]])
    # first player runs at the basket, second runs away from it
    out = motion_features(x, y, np.array([[5.0, -5.0]]), np.array([[0.0, 0.0]]), dist)

    assert out["v_radial"][0, 0] == pytest.approx(5.0)
    assert out["v_radial"][0, 1] == pytest.approx(-5.0)
    assert out["v_tangent"][0, 0] == pytest.approx(0.0, abs=1e-9)


def test_tangential_velocity_is_signed_around_the_basket():
    x = np.array([[HOOP[0] - 10.0, HOOP[0] - 10.0]])
    y = np.array([[25.0, 25.0]])
    dist = np.array([[10.0, 10.0]])
    out = motion_features(x, y, np.array([[0.0, 0.0]]), np.array([[4.0, -4.0]]), dist)
    assert out["v_tangent"][0, 0] == pytest.approx(-out["v_tangent"][0, 1])
    assert out["v_tangent"][0, 0] != 0


def test_extrapolation_damping_scales_the_step():
    x = np.array([[10.0]])
    y = np.array([[25.0]])
    flight = np.array([[2.0]])
    full = extrapolate(x, y, np.array([[3.0]]), np.array([[0.0]]), flight, damping=1.0)
    half = extrapolate(x, y, np.array([[3.0]]), np.array([[0.0]]), flight, damping=0.5)

    assert full["ext_x"][0, 0] == pytest.approx(16.0)
    assert half["ext_x"][0, 0] == pytest.approx(13.0)
    # moving toward the basket from x=10 closes the distance
    assert half["ext_closing"][0, 0] > 0


# --------------------------------------------------------------------------- #
# Shot context
# --------------------------------------------------------------------------- #


def test_bearing_is_measured_relative_to_the_shooter():
    angle = np.array([[0.4, 0.4, 1.4, -0.6]])
    is_shooter = np.array([[1.0, 0.0, 0.0, 0.0]])
    dist = np.ones((1, 4)) * 10
    out = shot_context(dist, angle, is_shooter, np.full((1, 4), 20.0))

    assert out["rel_bearing"][0, 0] == pytest.approx(0.0)
    assert out["rel_bearing"][0, 1] == pytest.approx(0.0)
    assert out["rel_bearing"][0, 2] == pytest.approx(1.0)
    # the two sides of the shot get opposite signs, which a cosine cannot express
    assert out["sin_shooter"][0, 2] > 0
    assert out["sin_shooter"][0, 3] < 0


def test_bearing_wraps_at_pi():
    """A player 10 degrees the other side of the baseline is 20 degrees away, not 340."""
    angle = np.array([[np.pi - 0.17, -np.pi + 0.17]])
    is_shooter = np.array([[1.0, 0.0]])
    out = shot_context(np.ones((1, 2)), angle, is_shooter, np.full((1, 2), 20.0))
    assert abs(out["rel_bearing"][0, 1]) == pytest.approx(0.34, abs=1e-6)


# --------------------------------------------------------------------------- #
# Fitted priors
# --------------------------------------------------------------------------- #


def test_priors_fit_and_transform_round_trip(frame):
    priors = ShotPriors().fit(frame)
    out = priors.transform(frame)

    assert len(out) == len(frame)
    for name in CONTEST_DERIVED:
        assert name in out.columns
        assert out[name].notna().all()
    assert np.isfinite(priors.damping_)


def test_transform_never_reads_the_measured_flight_time(frame):
    """The served feature is predicted from the shot, so corrupting the measured
    column must not move it. That column does not exist at serving time."""
    priors = ShotPriors().fit(frame)
    baseline = priors.transform(frame)

    corrupted = frame.copy()
    corrupted["FlightTime"] = 99.0
    after = priors.transform(corrupted)

    assert np.allclose(baseline["flight_hat"], after["flight_hat"])
    assert np.allclose(baseline["ext_x"], after["ext_x"])


def test_priors_are_fitted_only_on_what_they_are_given(frame):
    """Two disjoint fits must differ, or the fit is not using its input."""
    first = ShotPriors().fit(frame[frame["GameID"].isin(["g0", "g1"])])
    second = ShotPriors().fit(frame[frame["GameID"].isin(["g2", "g3"])])
    assert first.damping_ != second.damping_


def test_transform_preserves_canonical_slot_order(frame):
    shuffled = frame.sample(frac=1.0, random_state=0)
    out = ShotPriors().fit(frame).transform(shuffled)
    for _, group in out.groupby("ShotID", sort=False):
        assert list(group["Slot"]) == list(range(N_PLAYERS))


def test_served_features_need_no_velocity():
    """The web app collects positions, not direction vectors. Nothing in the served
    set may depend on one."""
    velocity_columns = {"pre_vx", "pre_vy", "pre_speed"}
    assert not velocity_columns & set(SERVED_FEATURES)
    assert not velocity_columns & set(STATIC_FEATURES)
    assert velocity_columns < set(RELEASE_FEATURES)
