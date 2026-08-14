"""Train/validation/test splitting.

The split is **by game and chronological**, and both halves of that matter.

*By game*, because two shots from the same game share the lineups on the floor, the
arena's tracking calibration, and the court itself. Splitting by shot puts near
duplicates on both sides of the boundary and inflates every number measured
afterwards -- the model gets credit for recognising a game it has already seen
rather than for generalising.

*Chronological*, because the intended use is predicting a rebound that has not
happened yet. A random split lets the model train on January and test on November,
which is a strictly easier problem than the one being solved and quietly launders
away any drift across the season.

Ordering is by ``GameID``. The NBA assigns these sequentially in schedule order
within a season -- this corpus runs contiguously from ``0021500001`` to
``0021500663`` -- so sorting the ids sorts the games by date. The one wrinkle is a
postponed game, which keeps the id it was assigned and is played later than its
neighbours; at this granularity that shifts a single game by days inside a split
containing a hundred, so it is noted rather than corrected.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Split:
    """One partition of the frame, kept alongside the games that produced it."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def summary(self) -> str:
        lines = []
        for name in ("train", "val", "test"):
            part = getattr(self, name)
            games = part["GameID"].nunique()
            shots = part["ShotID"].nunique()
            lines.append(f"{name:5}: {games:4} games  {shots:6,} shots  {len(part):7,} rows")
        return "\n".join(lines)


def split_by_game(
    rows: pd.DataFrame,
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    drop_team_rebounds: bool = True,
) -> Split:
    """Partition feature rows into train/val/test along game boundaries.

    ``fractions`` applies to the *number of games*, not the number of shots, so the
    boundaries fall between games and no shot is divided from its own ten rows.

    Team rebounds are dropped by default. They carry an all-zero label, which a
    softmax over the ten players cannot express: there is no correct slot to pick.
    Keeping them would either need an eleventh "nobody" outcome or would silently
    train the model toward a target that is never attainable. They are 8.8% of
    paired shots and are a genuinely different event, so the first pass excludes
    them and says so.
    """
    if not abs(sum(fractions) - 1.0) < 1e-9:
        raise ValueError(f"fractions must sum to 1, got {fractions} summing to {sum(fractions)}")

    if drop_team_rebounds:
        rows = rows[~rows["IsTeamRebound"].astype(bool)]

    games = sorted(rows["GameID"].unique())
    n = len(games)
    n_train = int(n * fractions[0])
    n_val = int(n * (fractions[0] + fractions[1]))

    boundaries = {
        "train": set(games[:n_train]),
        "val": set(games[n_train:n_val]),
        "test": set(games[n_val:]),
    }
    parts = {name: rows[rows["GameID"].isin(ids)] for name, ids in boundaries.items()}
    return Split(train=parts["train"], val=parts["val"], test=parts["test"])
