"""Play-by-play extraction: pair each missed shot with the rebound that followed.

Replaces ``PbP_extractor.py``. Changes beyond the Python 3 port, each of which is
covered in ``tests/test_pbp.py``:

* This module no longer fetches anything. The original ``requests.get`` against
  ``http://stats.nba.com`` was first replaced by a properly-headed, rate-limited,
  disk-cached client, and then by nothing at all: that host stopped answering
  entirely, holding connections open until they time out rather than refusing them,
  and it does so for a logged-in browser exactly as for a script. What survives is
  :func:`fetch` reading the disk cache, which holds one game. Live events come from
  :mod:`rebounding.data.bref` and arrive in the same shape :func:`to_frame` emits.
* The shot description is no longer dropped for visiting teams. The old code kept
  ``HOMEDESCRIPTION`` only, which left 44% of shots in the sample game with no text
  at all. Note it is not a plain coalesce: on a blocked shot *both* fields are
  populated, one with the shot and one with the block, so the rule is to take
  whichever field reports the miss.
* The rebound is found by searching forward for the next ``EVENTMSGTYPE == 4``
  rather than assuming it sits at ``shot_index + 1``. The old assumption held in all
  106 misses of the sample game, so this is defensive rather than a live fix.
* Team rebounds are kept and flagged instead of being silently dropped downstream.
  They are 11% of rebounds in the sample game.
* Row selection is positional throughout. The old code captured ``df.index`` before
  an in-place ``sort_values`` and then applied it via ``.iloc`` afterwards, mixing
  label space with position space. That was harmless only because the API already
  returns rows in the sorted order.
* Shot distance is parsed out of the description. Nothing in the original pipeline
  could tell a correctly paired shot from a mispaired one; comparing this against
  the tracking-derived distance is what makes pairing quality measurable.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from rebounding.constants import EVENT_REBOUND

# Events that can legitimately sit between a miss and its rebound:
# foul, violation, substitution, timeout, jump ball, ejection, instant replay.
SKIPPABLE_EVENTS = frozenset({6, 7, 8, 9, 10, 11, 18})

# A rebound more than this many seconds after the miss is not that miss's rebound.
MAX_REBOUND_GAP_SECONDS = 5

KEEP_COLUMNS = [
    "EVENTMSGTYPE",
    "EVENTMSGACTIONTYPE",
    "PERIOD",
    "PCTIMESTRING",
    "HOMEDESCRIPTION",
    "VISITORDESCRIPTION",
    "PLAYER1_ID",
    "PLAYER1_NAME",
    "PLAYER1_TEAM_ID",
    "PLAYER1_TEAM_ABBREVIATION",
    "PLAYER3_ID",
    "PLAYER3_NAME",
]

# Both source dialects: the NBA feed writes "Scola 5' Driving Floating Jump Shot",
# Basketball-Reference writes "misses 2-pt jump shot from 5 ft". See
# rebounding.data.bref for why the second one is now the source in practice.
_DISTANCE_RE = re.compile(r"(\d+)\s*(?:'|ft\b)")


def clock_to_seconds(clock: str) -> int:
    """``"MM:SS"`` to seconds remaining in the period."""
    minutes, seconds = clock.split(":")
    return 60 * int(minutes) + int(seconds)


def shot_description(home: str | None, visitor: str | None) -> str | None:
    """Return the description that reports the shot, not the block.

    On a blocked shot both fields are populated: the shooting team's side carries
    ``"MISS ..."`` and the defending team's carries ``"... BLOCK"``.
    """
    # JSON nulls arrive as float NaN once pandas has built the column.
    home = home if isinstance(home, str) else None
    visitor = visitor if isinstance(visitor, str) else None
    for text in (home, visitor):
        if text and text.startswith("MISS"):
            return text
    return home or visitor or None


def parse_shot_distance(description: str | None) -> float | None:
    """Feet, from text like ``"MISS Scola 5' Driving Floating Jump Shot"``.

    Returns ``None`` for descriptions without a distance. In the NBA dialect that is
    normal -- dunks and tip-ins are written without one. Basketball-Reference states
    a distance on those too, so the miss rate is far lower on that source.
    """
    if not isinstance(description, str):
        return None
    match = _DISTANCE_RE.search(description)
    return float(match.group(1)) if match else None


def fetch(game_id: str, cache_dir: str | Path | None = None) -> dict:
    """Raw play-by-play JSON for a game, from the disk cache.

    Cache-only by design. There is no longer a live source to fall back to, and a
    fallback that cannot work is worse than none: it would turn a missing game into
    a timeout thirty seconds later instead of an immediate, legible error. Use
    :func:`rebounding.data.bref.load` for games the cache does not hold, which in
    practice is all but one of them.
    """
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"{game_id}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text())

    raise FileNotFoundError(
        f"no cached play-by-play for game {game_id} under {cache_dir!r}. "
        "stats.nba.com no longer serves this data; use rebounding.data.bref instead."
    )


def to_frame(payload: dict) -> pd.DataFrame:
    """Raw API payload to a typed, chronologically ordered DataFrame."""
    result = payload["resultSets"][0]
    df = pd.DataFrame(result["rowSet"], columns=result["headers"])
    df = df[[c for c in KEEP_COLUMNS if c in df.columns]].copy()

    df["EVENTMSGTYPE"] = df["EVENTMSGTYPE"].astype(int)
    df["PERIOD"] = df["PERIOD"].astype(int)
    # Nullable Int64 rather than the old `.astype(str).str[:-2]`, which stripped the
    # ".0" off a float repr and would corrupt any id that did not have one.
    # The feed uses 0, not null, for an absent player -- PLAYER3_ID is 0 on every
    # unblocked shot -- so 0 has to become NA or `notna()` is true everywhere.
    for col in ("PLAYER1_ID", "PLAYER1_TEAM_ID", "PLAYER3_ID"):
        if col in df:
            numeric = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            df[col] = numeric.mask(numeric == 0)

    df["GameClock"] = df["PCTIMESTRING"].map(clock_to_seconds)
    df["Description"] = [
        shot_description(h, v) for h, v in zip(df["HOMEDESCRIPTION"], df["VISITORDESCRIPTION"], strict=True)
    ]

    # Sort first, then reset, so every later index is positional and unambiguous.
    df = df.sort_values(by=["PERIOD", "GameClock"], ascending=[True, False], kind="stable")
    return df.reset_index(drop=True)


def made_shots(df: pd.DataFrame) -> pd.DataFrame:
    """Period and clock of every made field goal.

    Pairing needs these even though they never become training rows: a make
    produces a ball-at-rim moment in the tracking data too, and if misses are
    allowed to claim those the assignment goes wrong.
    """
    made = df[df["EVENTMSGTYPE"] == 1]
    return made[["PERIOD", "GameClock"]].rename(
        columns={"PERIOD": "Period", "GameClock": "Clock"}
    ).reset_index(drop=True)


def find_rebound(df: pd.DataFrame, shot_pos: int) -> int | None:
    """Position of the rebound belonging to the miss at ``shot_pos``, or ``None``.

    Scans forward past administrative events. Gives up at any other event type, at
    a period boundary, or once more than :data:`MAX_REBOUND_GAP_SECONDS` have
    elapsed.
    """
    period = df.at[shot_pos, "PERIOD"]
    shot_clock = df.at[shot_pos, "GameClock"]

    for pos in range(shot_pos + 1, len(df)):
        if df.at[pos, "PERIOD"] != period:
            return None
        if shot_clock - df.at[pos, "GameClock"] > MAX_REBOUND_GAP_SECONDS:
            return None

        event = df.at[pos, "EVENTMSGTYPE"]
        if event == EVENT_REBOUND:
            return pos
        if event not in SKIPPABLE_EVENTS:
            return None
    return None


def pair_shots_and_rebounds(df: pd.DataFrame, game_id: str) -> pd.DataFrame:
    """One row per missed shot that has an identifiable rebound.

    Team rebounds are retained with ``IsTeamRebound`` set; on those rows
    ``RebPlayerID`` is null and ``RebTeamID`` carries the team. The NBA feed encodes
    a team rebound by putting the team id in ``PLAYER1_ID`` and leaving
    ``PLAYER1_TEAM_ID`` null.
    """
    misses = df.index[df["EVENTMSGTYPE"] == 2].to_numpy()

    records = []
    for shot_pos in misses:
        reb_pos = find_rebound(df, int(shot_pos))
        if reb_pos is None:
            continue

        shot = df.loc[shot_pos]
        reb = df.loc[reb_pos]
        is_team = pd.isna(reb["PLAYER1_TEAM_ID"])
        description = shot["Description"]

        records.append(
            {
                "GameID": game_id,
                "Period": shot["PERIOD"],
                "Clock": shot["GameClock"],
                "RebClock": reb["GameClock"],
                "ShootPlayerID": shot["PLAYER1_ID"],
                "ShootPlayerName": shot["PLAYER1_NAME"],
                "ShootTeamID": shot["PLAYER1_TEAM_ID"],
                "ShootTeamName": shot["PLAYER1_TEAM_ABBREVIATION"],
                "RebPlayerID": pd.NA if is_team else reb["PLAYER1_ID"],
                "RebPlayerName": None if is_team else reb["PLAYER1_NAME"],
                # For a team rebound the team id lives in PLAYER1_ID.
                "RebTeamID": reb["PLAYER1_ID"] if is_team else reb["PLAYER1_TEAM_ID"],
                "RebTeamName": reb["PLAYER1_TEAM_ABBREVIATION"],
                "IsTeamRebound": bool(is_team),
                "BlockPlayerID": shot["PLAYER3_ID"],
                "Description": description,
                "ShotDistance": parse_shot_distance(description),
            }
        )

    paired = pd.DataFrame.from_records(records)
    if paired.empty:
        return paired

    # Delimited so period 1 + clock 543 cannot collide with period 15 + clock 43,
    # which the old undelimited concatenation allowed.
    paired["ShotID"] = (
        paired["GameID"].astype(str)
        + "-"
        + paired["Period"].astype(str)
        + "-"
        + paired["Clock"].astype(str)
        + "-"
        + paired["ShootPlayerID"].astype(str)
    )
    return paired.drop_duplicates(subset="ShotID").reset_index(drop=True)


def load(game_id: str, cache_dir: str | Path | None = None) -> pd.DataFrame:
    """Fetch, parse and pair in one call."""
    return pair_shots_and_rebounds(to_frame(fetch(game_id, cache_dir)), game_id)
