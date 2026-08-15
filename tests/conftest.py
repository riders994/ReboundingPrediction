import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_GAME_ID = "0021500492"  # 01.01.2016 CHA at TOR
SAMPLE_BREF_SLUG = "201601010TOR"  # the same game on Basketball-Reference


@pytest.fixture(scope="session")
def pbp_cache_dir() -> Path:
    return FIXTURES / "pbp"


@pytest.fixture(scope="session")
def sample_game_id() -> str:
    return SAMPLE_GAME_ID


@pytest.fixture(scope="session")
def sportvu_fixture_path() -> Path:
    """Four consecutive events from the sample game, each containing a rim contact."""
    return FIXTURES / "sportvu" / f"{SAMPLE_GAME_ID}.json"


@pytest.fixture(scope="session")
def bref_html() -> str:
    """The Basketball-Reference play-by-play page for the sample game."""
    return (FIXTURES / "bref" / f"{SAMPLE_BREF_SLUG}.html").read_text(encoding="utf8")


@pytest.fixture(scope="session")
def sample_roster(sportvu_fixture_path) -> tuple[dict, dict, str, str]:
    """``(players, team_ids, home_abbrev, away_abbrev)`` for the sample game."""
    from rebounding.data import sportvu

    return sportvu.game_roster(json.loads(sportvu_fixture_path.read_text()))
