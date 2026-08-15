"""Tests for per-shot feature construction."""

import numpy as np
import pandas as pd
import pytest

from rebounding.data import features, pairing, pbp, sportvu

N_PLAYERS = features.N_PLAYERS


@pytest.fixture(scope="module")
def rows(pbp_cache_dir, sample_game_id, sportvu_fixture_path):
    raw = pbp.to_frame(pbp.fetch(sample_game_id, cache_dir=pbp_cache_dir))
    shots = pbp.pair_shots_and_rebounds(raw, sample_game_id)
    tracking = sportvu.load(sportvu_fixture_path)
    paired, _ = pairing.pair(tracking, shots, pbp.made_shots(raw))
    return features.build(tracking, paired)


def _synthetic_shot(tracking, *, reb_player, is_team, release=5, rim=20):
    """A shot Series over the fixture's tracking, with a chosen rebound outcome."""
    team_ids, player_ids = tracking.team_ids[release], tracking.player_ids[release]
    # Pick a team that actually has five players on the floor at this moment.
    values, counts = np.unique(team_ids, return_counts=True)
    shoot_team = values[np.argmax(counts == 5)]
    shooter = player_ids[team_ids == shoot_team][0]

    return pd.Series(
        {
            "ReleaseIdx": release, "RimIdx": rim, "Basket": "right", "Period": 1,
            "ShootTeamID": shoot_team, "ShootPlayerID": shooter,
            "RebPlayerID": reb_player, "IsTeamRebound": is_team, "FlightTime": 1.0,
            "ShotID": "synthetic", "GameID": "g",
        }
    )


class TestBoxgen:
    def test_counts_nearest_opponents(self):
        """Five defenders stacked on one attacker all count against that attacker."""
        offense = np.array([[10.0, 25.0], [30.0, 5.0], [30.0, 45.0], [20.0, 10.0], [20.0, 40.0]])
        defense = np.tile([10.5, 25.0], (5, 1))
        counts = features.boxgen(np.vstack([offense, defense]))

        assert counts.shape == (N_PLAYERS,)
        assert counts[0] == 5  # every defender is nearest to attacker 0
        assert counts[1:5].sum() == 0
        assert counts[5:].sum() == 5

    def test_rejects_a_bad_shape_instead_of_miscounting(self):
        """The original sliced [:5]/[5:] unchecked and returned wrong counts."""
        with pytest.raises(ValueError, match=r"\(10, 2\)"):
            features.boxgen(np.zeros((9, 2)))
        with pytest.raises(ValueError):
            features.boxgen(np.zeros((10, 3)))


class TestCanonicalOrdering:
    def test_offense_first_then_defense(self, rows):
        for _, shot in rows.groupby("ShotID"):
            ordered = shot.sort_values("Slot")["is_offense"].to_numpy()
            assert ordered[:5].all()
            assert not ordered[5:].any()

    def test_each_team_block_is_sorted_by_release_distance(self, rows):
        """Slot meaning must be stable, or a flat tensor is incoherent across shots."""
        for _, shot in rows.groupby("ShotID"):
            ordered = shot.sort_values("Slot")
            offense = ordered.iloc[:5]["pre_dist"].to_numpy()
            defense = ordered.iloc[5:]["pre_dist"].to_numpy()
            assert (np.diff(offense) >= -1e-5).all()
            assert (np.diff(defense) >= -1e-5).all()

    def test_ordering_uses_release_not_rim_positions(self, rows):
        """The web app only has release positions, so ordering must not need rim ones."""
        by_rim_would_differ = False
        for _, shot in rows.groupby("ShotID"):
            ordered = shot.sort_values("Slot")
            if not (np.diff(ordered.iloc[:5]["pos_dist"].to_numpy()) >= -1e-5).all():
                by_rim_would_differ = True
                break
        assert by_rim_would_differ, "expected at least one shot where rim order differs"


class TestFeatureRows:
    def test_ten_rows_per_shot(self, rows):
        assert (rows.groupby("ShotID").size() == N_PLAYERS).all()

    def test_no_missing_values(self, rows):
        """The old left join let a player missing at one frame produce NaN features."""
        assert rows[features.PLAYER_FEATURES].isna().sum().sum() == 0

    def test_exactly_five_players_a_side(self, rows):
        assert (rows.groupby("ShotID")["is_offense"].sum() == 5).all()

    def test_one_shooter_per_shot(self, rows):
        assert (rows.groupby("ShotID")["is_shooter"].sum() == 1).all()

    def test_individual_rebounds_have_exactly_one_rebounder(self, rows):
        per_shot = rows.groupby("ShotID").agg(
            rebounders=("Rebounder", "sum"), team=("IsTeamRebound", "first")
        )
        assert (per_shot.loc[~per_shot["team"], "rebounders"] == 1).all()

    def test_team_rebounds_are_kept_with_no_individual_rebounder(self, sportvu_fixture_path):
        """The old features() returned [] whenever no individual rebounder matched.

        Built directly rather than fished out of the fixture, which is only four
        events long; the full game has eight such shots.
        """
        tracking = sportvu.load(sportvu_fixture_path)
        shot = _synthetic_shot(tracking, reb_player=pd.NA, is_team=True)

        frame = features.shot_features(tracking, shot)
        assert frame is not None
        assert len(frame) == N_PLAYERS
        assert frame["IsTeamRebound"].all()
        assert frame["Rebounder"].sum() == 0

    def test_a_credited_rebounder_who_is_not_on_the_floor_is_rejected(self, sportvu_fixture_path):
        tracking = sportvu.load(sportvu_fixture_path)
        shot = _synthetic_shot(tracking, reb_player=999999, is_team=False)
        assert features.shot_features(tracking, shot) is None

    def test_velocity_is_present_and_physically_plausible(self, rows):
        """The old pipeline had no velocity at all -- two isolated frames only."""
        assert rows["pre_speed"].max() > 1.0
        # An NBA sprint tops out near 22 ft/s.
        assert rows["pre_speed"].max() < 35.0

    def test_positions_are_inside_the_half_court(self, rows):
        assert rows["pre_x"].between(-5, 52).all()
        assert rows["pre_y"].between(-5, 55).all()


class TestToTensor:
    def test_shape_and_label_alignment(self, rows):
        x, y, names = features.to_tensor(rows)
        n_shots = rows["ShotID"].nunique()
        assert x.shape == (n_shots, N_PLAYERS, len(names))
        assert y.shape == (n_shots, N_PLAYERS)

    def test_labels_match_the_long_rows(self, rows):
        x, y, _ = features.to_tensor(rows)
        expected = (
            rows.sort_values(["ShotID", "Slot"])["Rebounder"]
            .to_numpy()
            .reshape(-1, N_PLAYERS)
        )
        np.testing.assert_array_equal(y, expected)

    def test_rejects_incomplete_shots(self, rows):
        truncated = rows.iloc[:-1]
        with pytest.raises(ValueError, match="rows per shot"):
            features.to_tensor(truncated)


def test_shot_features_rejects_a_lineup_change_between_frames(sportvu_fixture_path):
    """A substitution or dropout between release and rim makes the shot unusable."""
    tracking = sportvu.load(sportvu_fixture_path)
    tracking.entity_ids[5, 3] = 999999  # break the lineup match at one frame

    shot = pd.Series(
        {
            "ReleaseIdx": 5, "RimIdx": 20, "Basket": "right", "Period": 1,
            "ShootTeamID": tracking.team_ids[20, 0], "ShootPlayerID": tracking.player_ids[20, 0],
            "RebPlayerID": pd.NA, "IsTeamRebound": True, "FlightTime": 1.0,
            "ShotID": "x", "GameID": "g",
        }
    )
    assert features.shot_features(tracking, shot) is None
