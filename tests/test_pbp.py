"""Tests for play-by-play extraction against the committed CHA-at-TOR game.

The counts asserted here were measured from the fixture and are what the old
pipeline was getting wrong.
"""

import pandas as pd
import pytest

from rebounding.data import pbp

# Measured from tests/fixtures/pbp/0021500492.json.
N_EVENTS = 448
N_MISSES = 106
N_REBOUNDS = 109
N_TEAM_REBOUNDS = 12
# Two of the 12 follow missed free throws (EVENTMSGTYPE 3), which we deliberately
# do not pair -- only field-goal misses become training rows.
N_TEAM_REBOUNDS_AFTER_FIELD_GOALS = 10
N_VISITOR_ONLY_DESCRIPTIONS = 47


@pytest.fixture(scope="module")
def raw(pbp_cache_dir, sample_game_id):
    return pbp.to_frame(pbp.fetch(sample_game_id, cache_dir=pbp_cache_dir))


@pytest.fixture(scope="module")
def paired(raw, sample_game_id):
    return pbp.pair_shots_and_rebounds(raw, sample_game_id)


def test_fetch_reads_from_cache_without_network(pbp_cache_dir, sample_game_id):
    payload = pbp.fetch(sample_game_id, cache_dir=pbp_cache_dir)
    assert payload["resultSets"][0]["rowSet"]


def test_fetch_raises_on_a_cache_miss_rather_than_reaching_for_a_dead_host(pbp_cache_dir):
    """There is no live NBA feed to fall back to; failing fast beats a timeout."""
    with pytest.raises(FileNotFoundError, match="no cached play-by-play"):
        pbp.fetch("0021500001", cache_dir=pbp_cache_dir)

    with pytest.raises(FileNotFoundError):
        pbp.fetch("0021500492", cache_dir=None)


def test_frame_shape_and_ordering(raw):
    assert len(raw) == N_EVENTS
    assert (raw["EVENTMSGTYPE"] == 2).sum() == N_MISSES
    assert (raw["EVENTMSGTYPE"] == 4).sum() == N_REBOUNDS
    # Chronological: period ascending, clock descending within period.
    keys = list(zip(raw["PERIOD"], -raw["GameClock"], strict=True))
    assert keys == sorted(keys)


def test_team_ids_are_nullable_ints_not_stringified_floats(raw):
    assert raw["PLAYER1_TEAM_ID"].dtype == "Int64"
    known = raw.loc[raw["PLAYER1_TEAM_ABBREVIATION"] == "TOR", "PLAYER1_TEAM_ID"].dropna()
    assert (known == 1610612761).all()


def test_visiting_team_shots_keep_their_description(raw):
    """The old code kept HOMEDESCRIPTION only, blanking 44% of shots."""
    misses = raw[raw["EVENTMSGTYPE"] == 2]
    visitor_only = misses["HOMEDESCRIPTION"].isna() & misses["VISITORDESCRIPTION"].notna()
    assert visitor_only.sum() == N_VISITOR_ONLY_DESCRIPTIONS
    assert misses["Description"].notna().all()


def test_blocked_shot_takes_the_shot_text_not_the_block_text():
    """Both fields are populated on a block; a plain coalesce picks the wrong one."""
    assert pbp.shot_description("MISS Scola 5' Driving Floating Jump Shot", "Zeller BLOCK (1 BLK)") == (
        "MISS Scola 5' Driving Floating Jump Shot"
    )
    # Same event with the teams the other way round.
    assert pbp.shot_description("Zeller BLOCK (1 BLK)", "MISS Scola 5' Driving Floating Jump Shot") == (
        "MISS Scola 5' Driving Floating Jump Shot"
    )


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        ("MISS Scola 5' Driving Floating Jump Shot", 5.0),
        ("MISS Lowry 26' 3PT Jump Shot", 26.0),
        ("MISS Biyombo Dunk", None),
        (None, None),
    ],
)
def test_parse_shot_distance(description, expected):
    assert pbp.parse_shot_distance(description) == expected


def test_every_miss_pairs_in_this_game(paired):
    assert len(paired) == N_MISSES


def test_team_rebounds_are_kept_and_flagged(paired, raw):
    """The old features() dropped these entirely -- 11% of rebounds."""
    all_team_rebounds = raw[(raw["EVENTMSGTYPE"] == 4) & raw["PLAYER1_TEAM_ID"].isna()]
    assert len(all_team_rebounds) == N_TEAM_REBOUNDS

    team = paired[paired["IsTeamRebound"]]
    assert len(team) == N_TEAM_REBOUNDS_AFTER_FIELD_GOALS
    assert team["RebPlayerID"].isna().all()
    # The team id lives in PLAYER1_ID on these rows and must survive.
    assert team["RebTeamID"].notna().all()

    individual = paired[~paired["IsTeamRebound"]]
    assert individual["RebPlayerID"].notna().all()


def test_shot_ids_are_unique_and_delimited(paired):
    assert paired["ShotID"].is_unique
    # Undelimited concatenation let period 1 + clock 543 collide with 15 + 43.
    assert paired["ShotID"].str.count("-").min() >= 3


def test_rebound_follows_the_shot_within_the_same_period(paired):
    assert (paired["RebClock"] <= paired["Clock"]).all()
    assert (paired["Clock"] - paired["RebClock"] <= pbp.MAX_REBOUND_GAP_SECONDS).all()


def test_shot_distance_is_populated_for_most_shots(paired):
    assert paired["ShotDistance"].notna().mean() > 0.8


def test_find_rebound_skips_administrative_events():
    """A loose-ball foul between the miss and the rebound must not break pairing."""
    df = pd.DataFrame(
        {
            "EVENTMSGTYPE": [2, 6, 8, 4],  # miss, foul, substitution, rebound
            "PERIOD": [1, 1, 1, 1],
            "GameClock": [500, 499, 499, 498],
        }
    )
    assert pbp.find_rebound(df, 0) == 3


def test_find_rebound_gives_up_on_an_intervening_shot():
    df = pd.DataFrame(
        {
            "EVENTMSGTYPE": [2, 1, 4],  # miss, made shot, rebound
            "PERIOD": [1, 1, 1],
            "GameClock": [500, 499, 498],
        }
    )
    assert pbp.find_rebound(df, 0) is None


def test_find_rebound_respects_period_and_clock_bounds():
    crosses_period = pd.DataFrame(
        {"EVENTMSGTYPE": [2, 4], "PERIOD": [1, 2], "GameClock": [0, 720]}
    )
    assert pbp.find_rebound(crosses_period, 0) is None

    too_late = pd.DataFrame(
        {"EVENTMSGTYPE": [2, 4], "PERIOD": [1, 1], "GameClock": [500, 480]}
    )
    assert pbp.find_rebound(too_late, 0) is None


def test_find_rebound_does_not_run_off_the_end():
    """`rebs = shots + 1` could index past the frame if a game ended on a miss."""
    df = pd.DataFrame({"EVENTMSGTYPE": [2], "PERIOD": [1], "GameClock": [0]})
    assert pbp.find_rebound(df, 0) is None
