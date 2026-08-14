"""Build the training frame across many games.

Replaces ``massunpack.py``, whose two problems were structural rather than
incidental:

* ``self.run = time.time()`` overwrote the ``run`` **method** with a float, so an
  instance could only be used once.
* ``except Exception: pass`` swallowed every failure. Games disappeared from the
  output with nothing recorded but a count, so a run that lost a third of the
  season looked the same as one that lost nothing.

Here every game either contributes rows or appears in the failure log with the
exception that stopped it, and the per-game pairing reports are aggregated so the
yield of a rebuild can be inspected rather than assumed.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from rebounding.data import bref, features, pairing, pbp, sportvu

log = logging.getLogger(__name__)


def game_events(
    tracking: sportvu.GameTracking,
    pbp_cache: str | Path | None = None,
    bref_cache: str | Path | None = None,
) -> pd.DataFrame:
    """The NBA-shaped event frame for one game, from whichever source has it.

    A cached NBA payload wins when one exists, because that is the feed the pairing
    logic was written and measured against. Otherwise the events come from
    Basketball-Reference -- ``stats.nba.com`` no longer answers, so for all but the
    one cached game that is the only live source. Both paths produce the same
    columns; see :mod:`rebounding.data.bref`.
    """
    if pbp_cache is not None and (Path(pbp_cache) / f"{tracking.game_id}.json").exists():
        return pbp.to_frame(pbp.fetch(tracking.game_id, cache_dir=pbp_cache))

    return bref.load(
        tracking.date,
        tracking.home_abbrev,
        tracking.away_abbrev,
        tracking.players,
        tracking.team_id_by_abbrev,
        cache_dir=bref_cache,
    )


@dataclass
class BuildReport:
    """Outcome of a bulk build, game by game."""

    n_games: int = 0
    n_succeeded: int = 0
    n_rows: int = 0
    n_shots_seen: int = 0
    n_shots_paired: int = 0
    drops: Counter = field(default_factory=Counter)
    failures: dict[str, str] = field(default_factory=dict)
    seconds: float = 0.0

    def summary(self) -> str:
        pair_rate = self.n_shots_paired / self.n_shots_seen if self.n_shots_seen else 0.0
        lines = [
            f"games      : {self.n_succeeded}/{self.n_games} succeeded",
            f"shots      : {self.n_shots_paired}/{self.n_shots_seen} paired ({pair_rate:.1%})",
            f"rows       : {self.n_rows}",
            f"elapsed    : {self.seconds:.0f}s",
        ]
        if self.drops:
            lines.append("drop reasons:")
            lines += [f"  {reason}: {count}" for reason, count in self.drops.most_common()]
        if self.failures:
            lines.append(f"failed games ({len(self.failures)}):")
            lines += [f"  {game}: {err}" for game, err in list(self.failures.items())[:20]]
            if len(self.failures) > 20:
                lines.append(f"  ... and {len(self.failures) - 20} more")
        return "\n".join(lines)


def build_game(
    tracking_path: str | Path,
    cache_dir: str | Path | None = None,
    bref_cache: str | Path | None = None,
) -> tuple[pd.DataFrame, pairing.PairingReport]:
    """Feature rows for a single game, from its tracking archive."""
    tracking = sportvu.load(tracking_path)
    game_id = tracking.game_id

    raw = game_events(tracking, pbp_cache=cache_dir, bref_cache=bref_cache)
    shots = pbp.pair_shots_and_rebounds(raw, game_id)
    paired, report = pairing.pair(tracking, shots, pbp.made_shots(raw))
    if paired.empty:
        return pd.DataFrame(), report
    return features.build(tracking, paired), report


def build_many(
    tracking_paths: list[str | Path],
    cache_dir: str | Path | None = None,
    output: str | Path | None = None,
    bref_cache: str | Path | None = None,
) -> tuple[pd.DataFrame, BuildReport]:
    """Build across many games, recording why each failure failed."""
    report = BuildReport(n_games=len(tracking_paths))
    started = time.time()
    frames = []

    for path in tracking_paths:
        path = Path(path)
        try:
            rows, pair_report = build_game(path, cache_dir=cache_dir, bref_cache=bref_cache)
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            report.failures[path.name] = f"{type(exc).__name__}: {exc}"
            log.warning("failed to build %s: %s", path.name, exc)
            continue

        report.n_succeeded += 1
        report.n_shots_seen += pair_report.n_shots
        report.n_shots_paired += pair_report.n_paired
        report.drops.update(pair_report.drops)
        if not rows.empty:
            frames.append(rows)
            report.n_rows += len(rows)
        log.info("%s", pair_report)

    report.seconds = time.time() - started
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if output is not None and not combined.empty:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(output, index=False)

    return combined, report
