"""Tests for the Basketball-Reference play-by-play source.

The important test here is :func:`test_matches_the_nba_feed_shot_for_shot`. One
game -- the sample game -- has a cached payload from the NBA feed that predates
``stats.nba.com`` going dark, which makes it the only game where the replacement
source can be checked against the original. Everything else in the corpus is built
on the assumption that the two agree, so that assumption is asserted rather than
believed.

Both frames are compared row-aligned after sorting, which is valid because both
produce exactly 106 misses in the same order.
"""

import numpy as np
import pandas as pd
import pytest

from rebounding.data import bref, pbp

# Measured against tests/fixtures/pbp/0021500492.json, the NBA feed for this game.
N_MISSES = 106
N_TEAM_REBOUNDS = 10

# The two sources credit two of the 106 rebounds to different players. Both are
# same-team disagreements -- the team is identical, the individual is not -- so they
# are scoring differences between the feeds rather than a parsing fault. Pinned so
# that a real regression in name resolution, which would break far more than two,
# cannot hide behind them. See test_rebounder_disagreements_are_same_team.
N_REBOUNDER_DISAGREEMENTS = 2


@pytest.fixture(scope="module")
def bref_frame(bref_html, sample_roster):
    players, team_ids, home, away = sample_roster
    return bref.to_frame(bref_html, players, team_ids, home, away)


@pytest.fixture(scope="module")
def bref_paired(bref_frame, sample_game_id):
    return _sorted(pbp.pair_shots_and_rebounds(bref_frame, sample_game_id))


@pytest.fixture(scope="module")
def nba_paired(pbp_cache_dir, sample_game_id):
    raw = pbp.to_frame(pbp.fetch(sample_game_id, cache_dir=pbp_cache_dir))
    return _sorted(pbp.pair_shots_and_rebounds(raw, sample_game_id))


def _sorted(df):
    return df.sort_values(["Period", "Clock"], ascending=[True, False]).reset_index(drop=True)


def _agree(left, right):
    """Element-wise equality treating NA as equal to NA."""
    return np.array(
        [
            (pd.isna(a) and pd.isna(b)) or (not pd.isna(a) and not pd.isna(b) and a == b)
            for a, b in zip(left, right, strict=True)
        ]
    )


# --------------------------------------------------------------------------- #
# Locating a game
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("date", "home", "expected"),
    [
        ("2016-01-01", "TOR", "201601010TOR"),
        # Three franchises are abbreviated differently by the two sites. Getting
        # these wrong is a 404, not a wrong game, but it would silently drop games.
        ("2015-12-05", "CHA", "201512050CHO"),
        ("2016-01-14", "BKN", "201601140BRK"),
        ("2015-11-02", "PHX", "201511020PHO"),
    ],
)
def test_game_slug(date, home, expected):
    assert bref.game_slug(date, home) == expected


def test_slug_comes_from_the_tracking_json_not_the_filename(sample_roster):
    """Ten archives have a directory path mangled into the filename; the JSON is clean."""
    _players, _teams, home, away = sample_roster
    assert (home, away) == ("TOR", "CHA")
    assert bref.game_slug("2016-01-01", home) == "201601010TOR"


# --------------------------------------------------------------------------- #
# Parsing pieces
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("clock", "expected"),
    [
        ("12:00.0", 720),
        ("11:41.0", 701),
        ("0:00.0", 0),
        # Tenths round up, matching the NBA feed's whole-second convention.
        ("5:30.4", 331),
        ("5:30.9", 331),
    ],
)
def test_clock_to_seconds(clock, expected):
    assert bref.clock_to_seconds(clock) == expected


def test_clock_rejects_junk():
    with pytest.raises(ValueError, match="unparseable clock"):
        bref.clock_to_seconds("Start of 1st quarter")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("L. Scola misses 2-pt jump shot from 4 ft", 2),
        ("J. Valančiūnas makes 2-pt dunk at rim", 1),
        ("Offensive rebound by J. Valančiūnas", 4),
        ("Defensive rebound by Team", 4),
        # Free throws are checked before makes/misses, or they would be shots.
        ("K. Lowry makes free throw 1 of 2", 3),
        ("K. Lowry misses free throw 2 of 2", 3),
        ("Turnover by K. Lowry (bad pass)", 5),
        ("Shooting foul by C. Zeller (drawn by K. Lowry)", 6),
        ("J. Lamb enters the game for N. Batum", 8),
        ("Start of 1st quarter", 12),
        ("End of 4th quarter", 13),
    ],
)
def test_event_type(text, expected):
    assert bref.event_type(text) == expected


def test_unrecognised_text_is_not_treated_as_skippable():
    """find_rebound skips administrative events; an unknown must stop it instead.

    Skipping something unrecognised could pair a miss with a later shot's rebound.
    Stopping only loses that one shot, which is the cheaper mistake.
    """
    assert bref.event_type("something the parser has never seen") == 0
    assert 0 not in pbp.SKIPPABLE_EVENTS


@pytest.mark.parametrize(
    ("first", "last", "expected"),
    [
        ("Luis", "Scola", "scolalu"),
        ("Jonas", "Valanciunas", "valanjo"),
        ("Dennis", "Schroder", "schrode"),
        ("Tim", "Hardaway Jr.", "hardati"),
    ],
)
def test_bref_id_key_reproduces_their_id_stem(first, last, expected):
    assert bref.bref_id_key(first, last) == expected


def test_resolve_player_by_id_stem_and_by_accented_display_name(sample_roster):
    players, _teams, _home, _away = sample_roster
    index = bref.build_player_index(players)

    assert bref.resolve_player("scolalu01", "L. Scola", index) == 2449
    # The display name carries diacritics the tracking roster does not.
    assert bref.resolve_player("valanjo01", "J. Valančiūnas", index) == 202685
    # A stem that is not in the roster still resolves off the display name.
    assert bref.resolve_player("", "J. Valančiūnas", index) == 202685


def test_resolve_player_raises_rather_than_returning_null(sample_roster):
    """A null ShootPlayerID silently disables pairing's shooter-proximity check."""
    players, _teams, _home, _away = sample_roster
    index = bref.build_player_index(players)
    with pytest.raises(KeyError, match="cannot resolve"):
        bref.resolve_player("jordami01", "M. Jordan", index)


# Marcus and Markieff Morris collide on *both* key forms: the surname and forename
# initial are shared, and the id stem is first-five-of-surname plus first-two-of-
# forename, so both are "morrima". Basketball-Reference separates them only by the
# trailing number. They were on opposing teams throughout 2015-16, so a DET-PHX
# game in the corpus puts both on the same page.
MORRIS_TWINS = {1: ("Marcus", "Morris", 100), 2: ("Markieff", "Morris", 200)}


def test_identically_keyed_players_do_not_resolve_game_wide():
    index = bref.build_player_index(MORRIS_TWINS)
    assert bref.bref_id_key("Marcus", "Morris") == bref.bref_id_key("Markieff", "Morris")
    assert index == {}
    with pytest.raises(KeyError, match="cannot resolve"):
        bref.resolve_player("morrima01", "M. Morris", index)


def test_identically_keyed_players_resolve_within_their_own_team():
    """Which column the event sits in is what breaks the tie."""
    detroit = bref.build_player_index(MORRIS_TWINS, team_id=100)
    phoenix = bref.build_player_index(MORRIS_TWINS, team_id=200)
    game_wide = bref.build_player_index(MORRIS_TWINS)

    assert bref.resolve_player("morrima01", "M. Morris", detroit, game_wide) == 1
    assert bref.resolve_player("morrima02", "M. Morris", phoenix, game_wide) == 2


# --------------------------------------------------------------------------- #
# Parsing a page
# --------------------------------------------------------------------------- #


def test_every_event_on_the_page_is_classified(bref_frame):
    assert (bref_frame["EVENTMSGTYPE"] == 0).sum() == 0


def test_frame_is_chronological(bref_frame):
    keys = list(zip(bref_frame["PERIOD"], -bref_frame["GameClock"], strict=True))
    assert keys == sorted(keys)


def test_events_are_attributed_to_the_right_column(bref_frame):
    """The six-cell row is time | away | away score | score | home score | home."""
    tor = bref_frame[bref_frame["PLAYER1_TEAM_ABBREVIATION"] == "TOR"]
    assert (tor["PLAYER1_TEAM_ID"].dropna() == 1610612761).all()
    cha = bref_frame[bref_frame["PLAYER1_TEAM_ABBREVIATION"] == "CHA"]
    assert (cha["PLAYER1_TEAM_ID"].dropna() == 1610612766).all()


def test_team_rebounds_use_the_nba_encoding(bref_frame):
    """pair_shots_and_rebounds detects them by a null team id, so reproduce that."""
    rebounds = bref_frame[bref_frame["EVENTMSGTYPE"] == 4]
    team = rebounds[rebounds["Description"].str.contains("by Team")]
    assert len(team) > 0
    assert team["PLAYER1_TEAM_ID"].isna().all()
    # The team id has to survive in PLAYER1_ID or the rebound loses its team.
    assert team["PLAYER1_ID"].notna().all()
    assert set(team["PLAYER1_ID"]) <= {1610612761, 1610612766}


def test_blocked_shot_records_the_blocker_not_the_shooter(bref_frame):
    blocked = bref_frame[
        (bref_frame["EVENTMSGTYPE"] == 2) & bref_frame["Description"].str.contains("block by")
    ]
    assert len(blocked) > 0
    assert blocked["PLAYER3_ID"].notna().all()
    # Shooter and blocker are different people on different teams.
    assert (blocked["PLAYER1_ID"] != blocked["PLAYER3_ID"]).all()


def test_shot_distance_parses_the_feet_dialect(bref_frame):
    """pbp.parse_shot_distance had to learn "from 4 ft" alongside "4'"."""
    assert pbp.parse_shot_distance("L. Scola misses 2-pt jump shot from 4 ft") == 4.0
    assert pbp.parse_shot_distance("K. Lowry misses 3-pt jump shot from 26 ft") == 26.0
    assert pbp.parse_shot_distance("MISS Scola 5' Driving Floating Jump Shot") == 5.0

    misses = bref_frame[bref_frame["EVENTMSGTYPE"] == 2]
    parsed = misses["Description"].map(pbp.parse_shot_distance)
    assert parsed.notna().mean() > 0.95


# --------------------------------------------------------------------------- #
# Equivalence with the NBA feed
# --------------------------------------------------------------------------- #


def test_same_number_of_misses_paired(bref_paired, nba_paired):
    assert len(bref_paired) == len(nba_paired) == N_MISSES


def test_matches_the_nba_feed_shot_for_shot(bref_paired, nba_paired):
    """Shooter, team, block and team-rebound flag agree on every one of 106 misses."""
    for column in (
        "Period",
        "ShootPlayerID",
        "ShootTeamID",
        "RebTeamID",
        "IsTeamRebound",
        "BlockPlayerID",
    ):
        agreement = _agree(nba_paired[column], bref_paired[column])
        assert agreement.all(), f"{column}: {(~agreement).sum()} of {len(agreement)} disagree"


def test_clocks_agree_within_a_second(bref_paired, nba_paired):
    """The feeds differ by at most one second, well inside pairing's tolerance."""
    delta = (nba_paired["Clock"] - bref_paired["Clock"]).abs()
    assert delta.max() <= 1


def test_rebounder_disagreements_are_same_team(bref_paired, nba_paired):
    """The two feeds credit two rebounds differently; both stay within the team."""
    differs = ~_agree(nba_paired["RebPlayerID"], bref_paired["RebPlayerID"])
    assert differs.sum() == N_REBOUNDER_DISAGREEMENTS
    # Same team on both sides, so the offensive/defensive label is unaffected and
    # only the individual credit differs.
    assert _agree(nba_paired["RebTeamID"][differs], bref_paired["RebTeamID"][differs]).all()


def test_team_rebounds_match(bref_paired, nba_paired):
    assert bref_paired["IsTeamRebound"].sum() == nba_paired["IsTeamRebound"].sum() == N_TEAM_REBOUNDS


def test_shot_distance_is_stated_more_often_than_the_nba_feed(bref_paired, nba_paired):
    """B-Ref gives a distance on layups and dunks, which the NBA text omits.

    More shots carrying a stated distance means more shots whose pairing can be
    checked against the tracking data, which is the only pairing quality signal
    there is.
    """
    assert bref_paired["ShotDistance"].notna().sum() > nba_paired["ShotDistance"].notna().sum()

    both = pd.DataFrame({"nba": nba_paired["ShotDistance"], "bref": bref_paired["ShotDistance"]}).dropna()
    # Where both state one they agree to within a foot of rounding.
    assert (both["nba"] - both["bref"]).abs().max() <= 1.0


def test_fetch_reads_from_cache_without_network(tmp_path, bref_html):
    cached = tmp_path / "201601010TOR.html"
    cached.write_text(bref_html, encoding="utf8")
    assert bref.fetch("201601010TOR", cache_dir=tmp_path) == bref_html
