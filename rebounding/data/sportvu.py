"""SportVU tracking extraction.

Replaces ``SportVU_extractor.py``. A raw game is a 104 MB JSON of ~211k "moments",
each holding the ball plus ten players at 25 Hz.

Changes beyond the Python 3 port:

* Positions live in numpy arrays rather than a Python list inside a DataFrame cell
  per moment. The old layout held the entire game as nested Python lists in an
  object column, which turned a 104 MB file into multiple GB of process memory.
* Deduplication is on the moment's wall-clock timestamp, which is exact. SportVU
  events overlap heavily -- in the sample game 211,445 raw moments collapse to
  79,703 distinct ones. The old key was six float columns including ``BallX/Y/Z``;
  it worked, but over-merged 106 genuinely distinct moments in this game (0.13%)
  where the clock and ball position happened to repeat. Minor, but the timestamp is
  a single exact integer and there is no reason to compare floats for identity.
* ``NearRim`` is evaluated against both real rim positions instead of folding the
  ball's x with ``abs(x - 47)``, so the extractor can also report *which* basket
  the ball is at. That is what :mod:`rebounding.data.court` needs to fold a shot
  correctly, and it is not recoverable after an ``abs()``.
* Shift-based transition flags are computed per quarter. The old code shifted
  across the whole game, so the first row of each quarter was compared against the
  last row of the previous one.
* Transitions are booleans. The old code encoded them as ``clock`` on transition
  rows and ``0`` elsewhere, so a filter like ``RimStart >= t`` matched every row in
  the quarter whenever ``t == 0`` -- every shot in the final second of a period
  paired against the end of the period.
* Players with a missing or unlisted position no longer raise ``KeyError``.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from rebounding.constants import (
    BALL_ID,
    DEFAULT_POSITION,
    POSITION_MAP,
    RIM_HEIGHT,
    RIM_LEFT,
    RIM_RIGHT,
)
from rebounding.data.court import LEFT, RIGHT

# How close to a rim, in feet, the ball must be to count as "at the rim".
RIM_RADIUS_FT = 1.0
# ...and how high. Slightly under the rim to tolerate tracking noise.
RIM_MIN_HEIGHT_FT = 9.9

# The looser "the shot has arrived" test: the ball descending through rim height
# somewhere around the basket. The tight NearRim test above only fires when the
# ball passes within a foot of the rim *centre*, which misses backboard caroms and
# hard rattles -- 20 of 95 unblocked misses in the sample game never trigger it.
# Descending through the rim plane near the basket covers 90 of those 95.
RIM_ARRIVAL_RADIUS_FT = 8.0

N_ENTITIES = 11  # ball + ten players


@dataclass
class GameTracking:
    """One game of tracking data.

    ``moments`` has one row per distinct moment with the clock and derived flags.
    The ``entity_*`` arrays are aligned to it positionally and span all eleven
    entities, with index 0 always the ball. The ``player_*`` properties drop the
    ball so that every player array shares one indexing scheme -- mixing the two
    is an easy off-by-one.
    """

    game_id: str
    moments: pd.DataFrame
    xyz: np.ndarray  # (n_moments, 11, 3) float32, ball at index 0
    entity_ids: np.ndarray  # (n_moments, 11) int64
    entity_team_ids: np.ndarray  # (n_moments, 11) int64
    roles: dict[str, float]  # player id -> ordinal listed position

    # Game identity and rosters, carried through because the play-by-play source is
    # now Basketball-Reference, which identifies a game by date and home team and
    # its players by name. See rebounding.data.bref.
    date: str = ""
    home_abbrev: str = ""
    away_abbrev: str = ""
    # Named for the key, not the value, because `team_ids` is already the per-moment
    # team array below and a dataclass field would shadow that property.
    team_id_by_abbrev: dict[str, int] = field(default_factory=dict)
    players: dict[int, tuple[str, str, int]] = field(default_factory=dict)

    @property
    def ball_xyz(self) -> np.ndarray:
        return self.xyz[:, 0, :]

    @property
    def player_xyz(self) -> np.ndarray:
        """``(n_moments, 10, 3)`` -- the ball dropped."""
        return self.xyz[:, 1:, :]

    @property
    def player_ids(self) -> np.ndarray:
        """``(n_moments, 10)``, aligned with :attr:`player_xyz`."""
        return self.entity_ids[:, 1:]

    @property
    def team_ids(self) -> np.ndarray:
        """``(n_moments, 10)``, aligned with :attr:`player_xyz`."""
        return self.entity_team_ids[:, 1:]

    def __len__(self) -> int:
        return len(self.moments)


def read_game_json(path: str | Path) -> dict:
    """Read a tracking game from ``.json`` or directly from its ``.7z`` archive."""
    path = Path(path)
    if path.suffix.lower() != ".7z":
        with open(path) as handle:
            return json.load(handle)

    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["7z", "x", "-y", f"-o{tmp}", str(path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        extracted = list(Path(tmp).glob("*.json"))
        if len(extracted) != 1:
            raise ValueError(f"expected exactly one json inside {path}, found {len(extracted)}")
        with open(extracted[0]) as handle:
            return json.load(handle)


def _roles(events: list[dict]) -> dict[str, float]:
    """Map player id to an ordinal listed position, from the first event's rosters.

    Done per game because rosters change across a season.
    """
    roles: dict[str, float] = {}
    for event in events:
        for side in ("home", "visitor"):
            for player in event.get(side, {}).get("players", []):
                roles[str(player["playerid"])] = POSITION_MAP.get(
                    player.get("position"), DEFAULT_POSITION
                )
        if roles:
            break
    return roles


def game_roster(payload: dict) -> tuple[dict[int, tuple[str, str, int]], dict[str, int], str, str]:
    """``(players, team_ids, home_abbrev, away_abbrev)`` from the first event.

    ``players`` maps NBA player id to ``(first, last, team_id)``. This is the only
    place the two teams' names and abbreviations exist in the corpus, and it is what
    lets :mod:`rebounding.data.bref` turn "L. Scola" back into an NBA player id.
    """
    players: dict[int, tuple[str, str, int]] = {}
    team_ids: dict[str, int] = {}
    home_abbrev = away_abbrev = ""

    for event in payload.get("events", []):
        for side in ("home", "visitor"):
            team = event.get(side)
            if not team:
                continue
            abbrev = str(team.get("abbreviation", ""))
            team_id = int(team["teamid"])
            team_ids[abbrev] = team_id
            if side == "home":
                home_abbrev = abbrev
            else:
                away_abbrev = abbrev
            for player in team.get("players", []):
                players[int(player["playerid"])] = (
                    str(player.get("firstname", "")),
                    str(player.get("lastname", "")),
                    team_id,
                )
        if players:
            break

    return players, team_ids, home_abbrev, away_abbrev


def unpack(payload: dict) -> GameTracking:
    """Raw tracking JSON to a :class:`GameTracking`."""
    events = payload["events"]
    game_id = str(payload.get("gameid", ""))

    quarters: list[int] = []
    timestamps: list[int] = []
    game_clocks: list[float] = []
    shot_clocks: list[float] = []
    entity_rows: list[list] = []

    seen: set[tuple[int, int]] = set()
    dropped_no_ball = 0

    for event in events:
        for moment in event["moments"]:
            quarter, timestamp, game_clock, shot_clock = moment[0], moment[1], moment[2], moment[3]
            key = (quarter, timestamp)
            if key in seen:
                continue

            entities = moment[5]
            # ~0.3% of moments lose the ball to a tracking dropout. Without the ball
            # there is no shot to find, so drop them rather than shifting the entity
            # axis and silently mislabelling a player as the ball.
            if len(entities) != N_ENTITIES or entities[0][0] != -1:
                dropped_no_ball += 1
                continue

            seen.add(key)
            quarters.append(quarter)
            timestamps.append(timestamp)
            game_clocks.append(game_clock)
            shot_clocks.append(shot_clock if shot_clock is not None else np.nan)
            entity_rows.append(entities)

    if not entity_rows:
        raise ValueError(f"no usable moments in game {game_id}")

    entities_arr = np.asarray(entity_rows, dtype=np.float64)  # (n, 11, 5)
    xyz = entities_arr[:, :, 2:5].astype(np.float32)
    team_ids = entities_arr[:, :, 0].astype(np.int64)
    player_ids = entities_arr[:, :, 1].astype(np.int64)

    moments = pd.DataFrame(
        {
            "Quarter": np.asarray(quarters, dtype=np.int16),
            "Timestamp": np.asarray(timestamps, dtype=np.int64),
            "GameClock": np.asarray(game_clocks, dtype=np.float32),
            "ShotClock": np.asarray(shot_clocks, dtype=np.float32),
        }
    )
    moments.attrs["dropped_no_ball"] = dropped_no_ball

    # Chronological. Timestamp is monotonic within a game and, unlike the truncated
    # integer clock the old code sorted on, has no ties.
    order = np.lexsort((moments["Timestamp"].to_numpy(), moments["Quarter"].to_numpy()))
    moments = moments.iloc[order].reset_index(drop=True)

    players, abbrev_to_team, home_abbrev, away_abbrev = game_roster(payload)

    tracking = GameTracking(
        game_id=game_id,
        moments=moments,
        xyz=xyz[order],
        entity_ids=player_ids[order],
        entity_team_ids=team_ids[order],
        roles=_roles(events),
        date=str(payload.get("gamedate", "")),
        home_abbrev=home_abbrev,
        away_abbrev=away_abbrev,
        team_id_by_abbrev=abbrev_to_team,
        players=players,
    )
    return add_ball_features(tracking)


def add_ball_features(tracking: GameTracking) -> GameTracking:
    """Add the rim-proximity and ball-direction flags used to locate shots."""
    moments = tracking.moments
    ball = tracking.ball_xyz

    # Distance to each real rim, on the full court. No abs() fold, so which basket
    # the ball is at survives -- court.fold needs it.
    d_left = np.hypot(ball[:, 0] - RIM_LEFT[0], ball[:, 1] - RIM_LEFT[1])
    d_right = np.hypot(ball[:, 0] - RIM_RIGHT[0], ball[:, 1] - RIM_RIGHT[1])
    nearest = np.minimum(d_left, d_right)

    near_rim = (nearest < RIM_RADIUS_FT) & (ball[:, 2] > RIM_MIN_HEIGHT_FT)
    moments["NearRim"] = near_rim
    moments["Basket"] = pd.Categorical(
        np.where(d_left < d_right, LEFT, RIGHT), categories=[LEFT, RIGHT]
    )
    moments["BallZ"] = ball[:, 2]
    moments["RimDistance"] = nearest

    # Per-quarter shifts. Shifting across the whole game compared the first row of
    # each quarter against the last row of the previous one.
    prev_z = moments.groupby("Quarter", sort=False)["BallZ"].shift()
    descending = (prev_z > moments["BallZ"]).fillna(False).astype(bool)
    moments["Descending"] = descending

    shifted = moments.groupby("Quarter", sort=False)[["NearRim", "Descending"]].shift()
    prev_near = shifted["NearRim"].fillna(False).astype(bool)
    prev_descending = shifted["Descending"].fillna(False).astype(bool)

    # Booleans, not clock-or-zero sentinels. The sentinel form made `RimStart >= 0`
    # match every row in the quarter for any shot at 0:00.
    moments["IsRimStart"] = near_rim & ~prev_near
    moments["IsHighStart"] = ~descending & prev_descending
    moments["IsLowStart"] = descending & ~prev_descending

    # The ball crossing the rim plane downward near the basket. This is what
    # pairing keys on; see RIM_ARRIVAL_RADIUS_FT.
    # NaN (the first row of a quarter) compares False, so no fill is needed.
    was_above_rim = (prev_z > RIM_HEIGHT).to_numpy(dtype=bool)
    moments["IsRimArrival"] = (
        (nearest < RIM_ARRIVAL_RADIUS_FT) & (ball[:, 2] <= RIM_HEIGHT) & was_above_rim
    )

    return tracking


def load(path: str | Path) -> GameTracking:
    """Read and unpack a tracking game from ``.json`` or ``.7z``."""
    return unpack(read_game_json(path))


def ball_id_matches(player_ids: np.ndarray) -> np.ndarray:
    """Boolean mask of ball entries, for callers working with raw id arrays."""
    return player_ids == int(BALL_ID)


def rim_height() -> float:
    return RIM_HEIGHT
