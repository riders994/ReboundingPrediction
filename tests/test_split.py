"""Tests for the game-level chronological split.

The failure this guards against is silent: a split that leaks produces *better*
numbers, so nothing looks wrong until the model is served and underperforms.
"""

import numpy as np
import pandas as pd
import pytest

from rebounding.eval.split import split_by_game

N_PLAYERS = 10


def make_frame(n_games=20, shots_per_game=5, team_rebound_every=7):
    """A frame with the columns the split cares about, ten rows per shot."""
    rng = np.random.default_rng(0)
    records = []
    shot_counter = 0
    for game in range(1, n_games + 1):
        game_id = f"00215{game:05d}"
        for _ in range(shots_per_game):
            shot_counter += 1
            is_team = shot_counter % team_rebound_every == 0
            winner = rng.integers(0, N_PLAYERS)
            for slot in range(N_PLAYERS):
                records.append(
                    {
                        "GameID": game_id,
                        "ShotID": f"{game_id}-{shot_counter}",
                        "Slot": slot,
                        "IsTeamRebound": is_team,
                        "Rebounder": 0 if is_team else int(slot == winner),
                        "pre_dist": float(rng.uniform(0, 40)),
                    }
                )
    return pd.DataFrame.from_records(records)


@pytest.fixture(scope="module")
def frame():
    return make_frame()


def test_fractions_must_sum_to_one(frame):
    with pytest.raises(ValueError, match="must sum to 1"):
        split_by_game(frame, fractions=(0.5, 0.3, 0.3))


def test_no_game_appears_in_two_splits(frame):
    split = split_by_game(frame)
    train, val, test = (set(p["GameID"]) for p in (split.train, split.val, split.test))
    assert not train & val
    assert not train & test
    assert not val & test


def test_split_is_chronological_not_random(frame):
    """Training on January to predict November is an easier problem than the real one."""
    split = split_by_game(frame)
    assert max(split.train["GameID"]) < min(split.val["GameID"])
    assert max(split.val["GameID"]) < min(split.test["GameID"])


def test_every_shot_keeps_all_ten_rows(frame):
    split = split_by_game(frame)
    for part in (split.train, split.val, split.test):
        assert (part.groupby("ShotID").size() == N_PLAYERS).all()


def test_team_rebounds_are_dropped_by_default(frame):
    """An all-zero label has no correct slot for a softmax over ten players to pick."""
    split = split_by_game(frame)
    for part in (split.train, split.val, split.test):
        assert not part["IsTeamRebound"].any()
        assert (part.groupby("ShotID")["Rebounder"].sum() == 1).all()


def test_team_rebounds_can_be_kept(frame):
    split = split_by_game(frame, drop_team_rebounds=False)
    kept = pd.concat([split.train, split.val, split.test])
    assert kept["IsTeamRebound"].any()
    assert len(kept) == len(frame)


def test_nothing_is_lost_or_duplicated(frame):
    split = split_by_game(frame, drop_team_rebounds=False)
    total = len(split.train) + len(split.val) + len(split.test)
    assert total == len(frame)


def test_fractions_apply_to_games_not_rows(frame):
    """Boundaries must fall between games, or a shot is divided from its own rows."""
    split = split_by_game(frame, fractions=(0.5, 0.25, 0.25), drop_team_rebounds=False)
    assert split.train["GameID"].nunique() == 10
    assert split.val["GameID"].nunique() == 5
    assert split.test["GameID"].nunique() == 5
