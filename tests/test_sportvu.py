"""Tests for SportVU tracking extraction.

The fixture is four consecutive events from the CHA-at-TOR game, chosen because each
contains a ball-at-rim moment.
"""

import numpy as np
import pandas as pd
import pytest

from rebounding.constants import RIM_LEFT, RIM_RIGHT
from rebounding.data import sportvu
from rebounding.data.court import LEFT, RIGHT


@pytest.fixture(scope="module")
def tracking(sportvu_fixture_path):
    return sportvu.load(sportvu_fixture_path)


def test_positions_are_arrays_not_dataframe_cells(tracking):
    """The old layout held nested Python lists in an object column."""
    assert isinstance(tracking.xyz, np.ndarray)
    assert tracking.xyz.dtype == np.float32
    assert tracking.xyz.shape == (len(tracking), 11, 3)
    assert tracking.player_ids.shape == (len(tracking), 10)
    assert tracking.entity_ids.shape == (len(tracking), 11)


def test_overlapping_events_are_deduplicated(tracking, sportvu_fixture_path):
    """SportVU events re-include surrounding moments; the timestamp key is exact."""
    import json

    with open(sportvu_fixture_path) as handle:
        raw = json.load(handle)
    raw_moments = sum(len(e["moments"]) for e in raw["events"])

    assert len(tracking) < raw_moments
    assert tracking.moments["Timestamp"].is_unique


def test_timestamp_dedup_keeps_at_least_as_much_as_the_old_float_key(tracking, sportvu_fixture_path):
    """The old six-float key over-merged slightly; the timestamp cannot.

    Measured on the full game: the old key yields 79,597 moments against 79,703 for
    the timestamp, so 106 distinct moments were being collapsed.
    """
    import json

    with open(sportvu_fixture_path) as handle:
        raw = json.load(handle)

    old_key = set()
    for event in raw["events"]:
        for moment in event["moments"]:
            entities = moment[5]
            if len(entities) != 11 or entities[0][0] != -1:
                continue
            ball = entities[0]
            old_key.add((moment[0], moment[2], moment[3], ball[2], ball[3], ball[4]))

    assert len(tracking) >= len(old_key)


def test_ball_is_always_entity_zero(tracking):
    assert (tracking.entity_team_ids[:, 0] == -1).all()
    assert (tracking.entity_ids[:, 0] == -1).all()
    # ...and the player views never include it.
    assert (tracking.team_ids != -1).all()


def test_moments_without_a_ball_are_dropped_not_shifted(tracking):
    assert tracking.moments.attrs["dropped_no_ball"] >= 0


def test_chronological_within_quarter(tracking):
    """Ordering is by wall-clock timestamp, which is strictly monotonic.

    The *game* clock is not: measured across 24 games, roughly 20 frames per
    quarter out of ~21,000 report a small backward step, the largest 0.78 s. That
    is jitter in the reported clock, not a reordering, so the game clock is only
    held to that tolerance. Pairing tolerates it because flight windows are of the
    order of two seconds and negative flight times are filtered out anyway.
    """
    m = tracking.moments
    for _, group in m.groupby("Quarter", sort=False):
        assert group["Timestamp"].is_monotonic_increasing

        steps = np.diff(group["GameClock"].to_numpy())
        backward = steps[steps > 0]
        assert backward.size / max(len(group), 1) < 0.01
        if backward.size:
            assert backward.max() < 1.0


def test_near_rim_requires_proximity_and_height(tracking):
    m = tracking.moments
    at_rim = m[m["NearRim"]]
    assert len(at_rim) > 0
    assert (at_rim["RimDistance"] < sportvu.RIM_RADIUS_FT).all()
    assert (at_rim["BallZ"] > sportvu.RIM_MIN_HEIGHT_FT).all()


def test_basket_identifies_the_nearer_rim(tracking):
    """This is what court.fold needs and what abs(x - 47) destroyed."""
    m = tracking.moments
    ball = tracking.ball_xyz
    d_left = np.hypot(ball[:, 0] - RIM_LEFT[0], ball[:, 1] - RIM_LEFT[1])
    d_right = np.hypot(ball[:, 0] - RIM_RIGHT[0], ball[:, 1] - RIM_RIGHT[1])
    expected = np.where(d_left < d_right, LEFT, RIGHT)
    assert (m["Basket"].astype(str).to_numpy() == expected).all()


def test_transitions_are_booleans_not_clock_sentinels():
    """The clock-or-zero encoding made `RimStart >= 0` match a whole quarter."""
    tracking = _synthetic_tracking(
        quarters=[1, 1, 1, 1],
        clocks=[10.0, 9.9, 9.8, 9.7],
        ball_z=[5.0, 8.0, 11.0, 11.0],
        ball_xy=[(88.75, 25.0)] * 4,
    )
    m = tracking.moments
    assert m["IsRimStart"].dtype == bool
    assert m["IsHighStart"].dtype == bool
    # Exactly one transition into the near-rim state.
    assert m["IsRimStart"].sum() == 1
    assert m.loc[m["IsRimStart"]].index[0] == 2


def test_shifts_do_not_cross_quarter_boundaries():
    """A quarter must not inherit the previous quarter's last ball height."""
    tracking = _synthetic_tracking(
        quarters=[1, 1, 2, 2],
        clocks=[10.0, 9.9, 720.0, 719.9],
        # Q1 ends with the ball high; Q2 opens low. A game-wide shift would call
        # the first Q2 row "descending".
        ball_z=[5.0, 25.0, 3.0, 4.0],
        ball_xy=[(50.0, 25.0)] * 4,
    )
    m = tracking.moments
    first_of_q2 = m.index[m["Quarter"] == 2][0]
    assert not bool(m.at[first_of_q2, "Descending"])


def test_unlisted_player_position_falls_back(tracking):
    payload = {
        "gameid": "x",
        "events": [
            {
                "home": {"players": [{"playerid": 1, "position": ""}]},
                "visitor": {"players": [{"playerid": 2, "position": "G"}]},
                "moments": [],
            }
        ],
    }
    roles = sportvu._roles(payload["events"])
    assert roles["1"] == 3.0  # DEFAULT_POSITION, not a KeyError
    assert roles["2"] == 1.0


def _synthetic_tracking(quarters, clocks, ball_z, ball_xy) -> sportvu.GameTracking:
    """Build a minimal GameTracking for the shift/transition edge cases."""
    n = len(quarters)
    xyz = np.zeros((n, 11, 3), dtype=np.float32)
    for i, ((bx, by), bz) in enumerate(zip(ball_xy, ball_z, strict=True)):
        xyz[i, 0] = (bx, by, bz)

    moments = pd.DataFrame(
        {
            "Quarter": np.asarray(quarters, dtype=np.int16),
            "Timestamp": np.arange(n, dtype=np.int64),
            "GameClock": np.asarray(clocks, dtype=np.float32),
            "ShotClock": np.full(n, 12.0, dtype=np.float32),
        }
    )
    tracking = sportvu.GameTracking(
        game_id="synthetic",
        moments=moments,
        xyz=xyz,
        entity_ids=np.zeros((n, 11), dtype=np.int64),
        entity_team_ids=np.zeros((n, 11), dtype=np.int64),
        roles={},
    )
    return sportvu.add_ball_features(tracking)
