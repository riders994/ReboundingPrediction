"""Play-by-play from Basketball-Reference, as a stand-in for the NBA stats API.

:mod:`rebounding.data.pbp` used to fetch from ``stats.nba.com/stats/*``. That host
no longer answers: the request is accepted and then held open until it times out,
with or without a browser's cookies, from a logged-in session as readily as from a
script. The sibling feeds are gone too --
``data.nba.net`` no longer resolves, and ``cdn.nba.com``'s live feed 403s for a
2015-16 game. The corpus this repo is built on is 2015-16, so the tracking data
outlived its play-by-play source.

Basketball-Reference still serves the same games. This module turns one of its
play-by-play pages into **the frame :func:`rebounding.data.pbp.to_frame` would have
returned**, so everything downstream -- ``pair_shots_and_rebounds``, ``made_shots``,
pairing, features -- runs unmodified against it. That is deliberate: the pairing
logic is the part that took work to get right, and re-implementing it against a
second source would double the surface area that can silently go wrong.

Two differences from the NBA feed are worth knowing:

* **Player identity has to be recovered.** Basketball-Reference writes "L. Scola"
  and links ``scolalu01``; everything downstream joins on NBA player ids. Both are
  resolved against the roster carried inside the tracking JSON for that same game,
  so the mapping is per game and bounded to ~26 players. An unresolvable or
  ambiguous name raises rather than silently producing a null id, because a null
  ``ShootPlayerID`` would quietly disable the shooter-proximity check in pairing.

* **Shot distance is stated more often.** The NBA text omits it on layups and dunks
  ("MISS Biyombo Dunk"); Basketball-Reference writes "misses 2-pt layup from 1 ft".
  Since pairing quality is measured by comparing this against the tracking-derived
  release distance, that is a gain -- more shots become checkable.

Equivalence against the NBA feed is asserted in ``tests/test_bref.py``, which
compares both sources over the one game for which a cached NBA payload survives.
"""

from __future__ import annotations

import re
import time
import unicodedata
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd

# Basketball-Reference asks for no more than 20 requests a minute and enforces it
# with a temporary ban rather than a 429. A 636-game rebuild is ~35 minutes at this
# rate, which is a one-off cost paid into a disk cache.
MIN_REQUEST_INTERVAL_S = 3.0

BASE_URL = "https://www.basketball-reference.com/boxscores/pbp"

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# NBA abbreviation -> Basketball-Reference abbreviation. Only the ones that differ.
BREF_TEAM_ABBREV = {"CHA": "CHO", "BKN": "BRK", "PHX": "PHO"}

_TIME_RE = re.compile(r"^(\d+):(\d+(?:\.\d+)?)$")
_QUARTER_ID_RE = re.compile(r"^q(\d+)$")
_BREF_PLAYER_RE = re.compile(r"/players/[a-z]/([a-z0-9']+)\.html")
_BLOCK_RE = re.compile(r"\(block by ", re.I)
_NON_ALPHA_RE = re.compile(r"[^a-z]")

# NBA EVENTMSGTYPE codes, reproduced so the frame is interchangeable with the one
# built from the NBA feed. pbp.SKIPPABLE_EVENTS is defined in terms of these.
_MADE_SHOT = 1
_MISSED_SHOT = 2
_FREE_THROW = 3
_REBOUND = 4
_TURNOVER = 5
_FOUL = 6
_VIOLATION = 7
_SUBSTITUTION = 8
_TIMEOUT = 9
_JUMP_BALL = 10
_EJECTION = 11
_PERIOD_BEGIN = 12
_PERIOD_END = 13
_UNKNOWN = 0

# Ordered: the first pattern that matches wins, so "misses free throw" is a free
# throw before it is a miss. Anything unmatched stays _UNKNOWN, which find_rebound
# treats as a hard stop -- the conservative direction, since skipping over a real
# event would pair a miss with someone else's rebound.
_EVENT_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"free throw", re.I), _FREE_THROW),
    (re.compile(r"\brebound by\b", re.I), _REBOUND),
    (re.compile(r"\bmakes\b", re.I), _MADE_SHOT),
    (re.compile(r"\bmisses\b", re.I), _MISSED_SHOT),
    (re.compile(r"\bturnover by\b", re.I), _TURNOVER),
    (re.compile(r"foul by\b", re.I), _FOUL),
    (re.compile(r"\bviolation\b", re.I), _VIOLATION),
    (re.compile(r"\benters the game for\b", re.I), _SUBSTITUTION),
    (re.compile(r"\btimeout\b", re.I), _TIMEOUT),
    (re.compile(r"^Jump ball", re.I), _JUMP_BALL),
    (re.compile(r"\bejected\b", re.I), _EJECTION),
    (re.compile(r"^Start of\b", re.I), _PERIOD_BEGIN),
    (re.compile(r"^End of\b", re.I), _PERIOD_END),
    (re.compile(r"instant replay", re.I), 18),
]

HOME = "home"
AWAY = "away"


# --------------------------------------------------------------------------- #
# Locating the page
# --------------------------------------------------------------------------- #


def game_slug(date: str, home_abbrev: str) -> str:
    """``("2016-01-01", "TOR")`` to ``"201601010TOR"``.

    Both inputs come out of the tracking JSON itself (``gamedate`` and the home
    team's ``abbreviation``), so no filename parsing is involved -- the ``.7z``
    names are unreliable anyway; ten of them have a directory path mangled in.
    """
    abbrev = home_abbrev.upper()
    return f"{date.replace('-', '')}0{BREF_TEAM_ABBREV.get(abbrev, abbrev)}"


def game_url(slug: str) -> str:
    return f"{BASE_URL}/{slug}.html"


_last_request_at = 0.0


def fetch(slug: str, cache_dir: str | Path | None = None) -> str:
    """Page HTML for one game, reading from cache when present.

    Network requests are spaced by :data:`MIN_REQUEST_INTERVAL_S`; cache hits are
    not, so a rebuild over a warm cache runs at full speed.
    """
    global _last_request_at

    cache_path = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"{slug}.html"
        if cache_path.exists():
            return cache_path.read_text(encoding="utf8")

    import requests

    wait = MIN_REQUEST_INTERVAL_S - (time.monotonic() - _last_request_at)
    if wait > 0:
        time.sleep(wait)

    response = requests.get(game_url(slug), headers={"User-Agent": USER_AGENT}, timeout=30)
    _last_request_at = time.monotonic()
    response.raise_for_status()
    html = response.text

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(html, encoding="utf8")
    return html


# --------------------------------------------------------------------------- #
# Player identity
# --------------------------------------------------------------------------- #


def _ascii_alpha(text: str) -> str:
    """Lowercase ASCII letters only: ``"Valančiūnas"`` -> ``"valanciunas"``."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return _NON_ALPHA_RE.sub("", stripped.lower())


def bref_id_key(first: str, last: str) -> str:
    """The identifying stem of a Basketball-Reference player id.

    Their ids are the first five letters of the surname plus the first two of the
    forename plus a disambiguating number -- ``Luis Scola`` -> ``scolalu01``. Taking
    the stem gives an exact key that does not depend on how the name is displayed.
    """
    return _ascii_alpha(last)[:5] + _ascii_alpha(first)[:2]


def display_key(first_initial: str, last: str) -> str:
    """Key for the displayed form, ``"L. Scola"`` -> ``"lscola"``."""
    return _ascii_alpha(first_initial)[:1] + _ascii_alpha(last)


def build_player_index(
    players: dict[int, tuple[str, str, int]], team_id: int | None = None
) -> dict[str, int]:
    """Lookup keys to NBA player id, over one team's roster or the whole game.

    ``players`` maps NBA player id to ``(first, last, team_id)``. Both key forms are
    registered. A key two players would share is dropped rather than resolved
    arbitrarily, so a collision surfaces as an unresolvable name in
    :func:`resolve_player` instead of a wrong id.

    Restricting to one team is what makes the Morris twins tractable: Marcus and
    Markieff Morris produce the same id stem (``morrima``) *and* the same display
    key, so neither resolves game-wide, but they were on opposing teams for the
    whole of this corpus and each is unique within his own roster.
    """
    index: dict[str, int] = {}
    collisions: set[str] = set()

    for player_id, (first, last, team) in players.items():
        if team_id is not None and team != team_id:
            continue
        for key in (bref_id_key(first, last), display_key(first, last)):
            if key in index and index[key] != player_id:
                collisions.add(key)
            index[key] = player_id

    for key in collisions:
        del index[key]
    return index


def resolve_player(bref_id: str, display_name: str, *indexes: dict[str, int]) -> int:
    """NBA player id for one linked name, trying each index in turn.

    Within an index the id stem is tried before the display name, the stem being the
    stronger key. Callers pass the index for the team whose column the event sits in
    first and the game-wide index second, so an ambiguous surname is resolved by the
    team that the event already tells us it belongs to.

    Raises rather than returning ``None``: downstream a missing ``ShootPlayerID``
    silently disables the shooter-proximity check that pairing depends on, so a name
    we cannot place has to stop the game rather than quietly degrade it.
    """
    stem = re.sub(r"\d+$", "", _ascii_alpha(bref_id))
    name = display_name.replace("\xa0", " ").strip()
    initial, _, surname = name.partition(".") if "." in name else ("", "", "")

    for index in indexes:
        if stem in index:
            return index[stem]
        if surname:
            key = display_key(initial, surname)
            if key in index:
                return index[key]

    raise KeyError(f"cannot resolve {display_name!r} (bref id {bref_id!r}) against the game roster")


# --------------------------------------------------------------------------- #
# Parsing the page
# --------------------------------------------------------------------------- #


class _Cell:
    __slots__ = ("attrs", "links", "parts")

    def __init__(self, attrs: dict[str, str]) -> None:
        self.attrs = attrs
        self.parts: list[str] = []
        self.links: list[tuple[str, str]] = []

    @property
    def text(self) -> str:
        return re.sub(r"\s+", " ", "".join(self.parts).replace("\xa0", " ")).strip()


class _PlayByPlayParser(HTMLParser):
    """Extracts the rows of ``<table id="pbp">`` with their cells and player links.

    Uses the stdlib parser rather than a regex over the markup, and rather than
    adding a scraping dependency: the row shape (six cells, or two when an event
    spans the table) is what the mapping below keys on, and a regex that loses a
    cell boundary would silently attribute events to the wrong team.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._in_table = False
        self._table_depth = 0
        self._cell: _Cell | None = None
        self._link: list[str] | None = None
        self.rows: list[tuple[int, list[_Cell]]] = []
        self._row: list[_Cell] | None = None
        self._quarter = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {k: (v or "") for k, v in attrs}

        if tag == "table":
            if self._in_table:
                self._table_depth += 1
            elif attr.get("id") == "pbp":
                self._in_table = True
                self._table_depth = 1
            return
        if not self._in_table:
            return

        if tag == "tr":
            match = _QUARTER_ID_RE.match(attr.get("id", ""))
            if match:
                self._quarter = int(match.group(1))
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = _Cell(attr)
        elif tag == "a" and self._cell is not None:
            self._link = [attr.get("href", ""), ""]

    def handle_endtag(self, tag: str) -> None:
        if not self._in_table:
            return

        if tag == "table":
            self._table_depth -= 1
            if self._table_depth == 0:
                self._in_table = False
        elif tag == "a" and self._link is not None and self._cell is not None:
            self._cell.links.append((self._link[0], self._link[1]))
            self._link = None
        elif tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(self._cell)
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append((self._quarter, self._row))
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._link is not None:
            self._link[1] += data
        if self._cell is not None:
            self._cell.parts.append(data)


def clock_to_seconds(clock: str) -> int:
    """``"11:41.0"`` to whole seconds remaining, rounding up.

    The NBA feed reports a whole-second ``PCTIMESTRING`` that is the ceiling of the
    true remaining time -- a shot with 11:40.4 left is stamped 11:41 -- while
    Basketball-Reference keeps the tenth. Rounding up reproduces the NBA convention,
    which matters because pairing matches these clocks against the tracking data.
    """
    match = _TIME_RE.match(clock.strip())
    if not match:
        raise ValueError(f"unparseable clock {clock!r}")
    minutes, seconds = match.group(1), float(match.group(2))
    whole = int(seconds)
    if seconds > whole:
        whole += 1
    return 60 * int(minutes) + whole


def event_type(text: str) -> int:
    for pattern, code in _EVENT_PATTERNS:
        if pattern.search(text):
            return code
    return _UNKNOWN


def _side_of(row: list[_Cell]) -> tuple[str | None, _Cell | None]:
    """Which team's column an event row sits in, and that column's cell.

    A six-cell row is ``time | away | away score | score | home score | home``. Rows
    that span the table (the jump ball, period boundaries) belong to neither side.
    """
    if len(row) == 2 and row[1].attrs.get("colspan"):
        return None, row[1]
    if len(row) != 6:
        return None, None
    away, home = row[1], row[5]
    if away.text:
        return AWAY, away
    if home.text:
        return HOME, home
    return None, None


def to_frame(
    html: str,
    players: dict[int, tuple[str, str, int]],
    team_ids: dict[str, int],
    home_abbrev: str,
    away_abbrev: str,
) -> pd.DataFrame:
    """Page HTML to the frame :func:`rebounding.data.pbp.to_frame` would produce.

    ``players`` maps NBA player id to ``(first, last, team_id)`` and ``team_ids``
    maps abbreviation to NBA team id; both come from the game's tracking JSON via
    :func:`rebounding.data.sportvu.game_roster`.
    """
    parser = _PlayByPlayParser()
    parser.feed(html)
    if not parser.rows:
        raise ValueError("no play-by-play table found in page")

    index = build_player_index(players)
    per_team = {tid: build_player_index(players, tid) for tid in set(team_ids.values())}
    abbrev_of = {HOME: home_abbrev, AWAY: away_abbrev}

    records = []
    for quarter, row in parser.rows:
        if not row or not _TIME_RE.match(row[0].text):
            continue  # header, or the quarter banner
        side, cell = _side_of(row)
        if cell is None:
            continue

        text = cell.text
        code = event_type(text)
        clock = clock_to_seconds(row[0].text)

        linked = [
            (match.group(1), label) for href, label in cell.links if (match := _BREF_PLAYER_RE.search(href))
        ]

        player1_id: int | None = None
        player1_name: str | None = None
        player1_team: int | None = None
        player3_id: int | None = None

        abbrev = abbrev_of.get(side) if side else None
        team_id = team_ids.get(abbrev) if abbrev else None

        # The event sits in one team's column, and every event's first player
        # belongs to that team, so that roster is the most specific place to look.
        own = per_team.get(team_id, {}) if team_id is not None else {}
        opponent = next((per_team[tid] for tid in per_team if tid != team_id), {})

        if linked:
            player1_id = resolve_player(linked[0][0], linked[0][1], own, index)
            player1_name = linked[0][1].strip()
            player1_team = team_id
            # "... misses 2-pt jump shot from 4 ft (block by C. Zeller)" -- the
            # blocker is the link after the parenthetical, and is by definition on
            # the other team, so his roster is searched first.
            if code == _MISSED_SHOT and _BLOCK_RE.search(text) and len(linked) > 1:
                player3_id = resolve_player(linked[1][0], linked[1][1], opponent, index)
        elif code == _REBOUND and team_id is not None:
            # A team rebound. The NBA feed encodes these by putting the team id in
            # PLAYER1_ID and leaving PLAYER1_TEAM_ID null; pair_shots_and_rebounds
            # detects them exactly that way, so reproduce it rather than inventing
            # a flag it would not read.
            player1_id = team_id
            player1_team = None

        records.append(
            {
                "EVENTMSGTYPE": code,
                "EVENTMSGACTIONTYPE": 0,
                "PERIOD": quarter,
                "PCTIMESTRING": f"{clock // 60}:{clock % 60:02d}",
                "HOMEDESCRIPTION": text if side == HOME else None,
                "VISITORDESCRIPTION": text if side == AWAY else None,
                "PLAYER1_ID": player1_id,
                "PLAYER1_NAME": player1_name,
                "PLAYER1_TEAM_ID": player1_team,
                "PLAYER1_TEAM_ABBREVIATION": abbrev,
                "PLAYER3_ID": player3_id,
                "PLAYER3_NAME": None,
                "GameClock": clock,
                "Description": text,
            }
        )

    df = pd.DataFrame.from_records(records)
    for col in ("PLAYER1_ID", "PLAYER1_TEAM_ID", "PLAYER3_ID"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df["EVENTMSGTYPE"] = df["EVENTMSGTYPE"].astype(int)
    df["PERIOD"] = df["PERIOD"].astype(int)

    df = df.sort_values(by=["PERIOD", "GameClock"], ascending=[True, False], kind="stable")
    return df.reset_index(drop=True)


def load(
    date: str,
    home_abbrev: str,
    away_abbrev: str,
    players: dict[int, tuple[str, str, int]],
    team_ids: dict[str, int],
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Fetch and parse one game, returning the NBA-shaped event frame."""
    html = fetch(game_slug(date, home_abbrev), cache_dir=cache_dir)
    return to_frame(html, players, team_ids, home_abbrev, away_abbrev)
