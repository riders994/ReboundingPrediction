"""Tests for locating play-by-play shots in the tracking data."""

import numpy as np
import pandas as pd
import pytest

from rebounding.data import pairing


def _moments(quarters, clocks):
    return pd.DataFrame(
        {
            "Quarter": np.asarray(quarters, dtype=np.int16),
            "GameClock": np.asarray(clocks, dtype=np.float32),
        }
    )


class TestAssignRimContacts:
    def test_matches_each_shot_to_its_nearest_contact(self):
        assignment = pairing.assign_rim_contacts(
            shot_clocks=np.array([500.0, 480.0]),
            rim_starts=np.array([10, 20]),
            rim_clocks=np.array([501.0, 481.0]),
        )
        assert assignment == {0: 10, 1: 20}

    def test_global_assignment_beats_greedy_claiming(self):
        """Greedy order loses a shot when an earlier one takes a shared candidate.

        Both shots are nearest to the same contact. Greedily, the first claims it
        and the second is dropped; the assignment gives each one.
        """
        assignment = pairing.assign_rim_contacts(
            shot_clocks=np.array([500.0, 499.0]),
            rim_starts=np.array([10, 20]),
            rim_clocks=np.array([499.5, 498.0]),
        )
        assert len(assignment) == 2
        assert set(assignment.values()) == {10, 20}

    def test_contacts_outside_the_window_are_not_matched(self):
        assignment = pairing.assign_rim_contacts(
            shot_clocks=np.array([500.0]),
            rim_starts=np.array([10]),
            rim_clocks=np.array([400.0]),  # 100 s away
        )
        assert assignment == {}

    def test_empty_inputs(self):
        assert pairing.assign_rim_contacts(np.array([]), np.array([]), np.array([])) == {}


class TestFindRelease:
    @staticmethod
    def _ball(z_values, rim_distance=25.0):
        n = len(z_values)
        xyz = np.zeros((n, 3), dtype=np.float32)
        xyz[:, 2] = z_values
        return xyz, np.full(n, rim_distance, dtype=np.float32)

    def test_rejects_rim_jitter_in_favour_of_the_real_release(self):
        """The failure that produced 24-foot threes three feet from the basket.

        Frame 2 is the release: the ball climbs from 3 ft to 16 ft. Frame 8 is the
        ball rattling at the rim -- also a "high start", also inside the flight
        bounds, but it rises less than a foot.
        """
        z = [4.0, 3.0, 3.0, 8.0, 14.0, 16.0, 12.0, 9.5, 9.6, 10.2, 9.8]
        ball, rim_dist = self._ball(z)
        moments = _moments([1] * 11, np.arange(11)[::-1] * 0.2 + 250)

        release = pairing.find_release(
            moments,
            high_starts=np.array([2, 8]),
            rim_idx=10,
            ball_xyz=ball,
            rim_distances=rim_dist,
        )
        assert release == 2

    def test_near_the_basket_a_small_rise_still_counts(self):
        """A layup releases near rim height and barely climbs."""
        z = [7.0, 7.0, 8.5, 10.8, 10.0]
        ball, rim_dist = self._ball(z, rim_distance=4.0)
        moments = _moments([1] * 5, np.array([251.0, 250.8, 250.6, 250.4, 250.2]))

        release = pairing.find_release(
            moments, np.array([1]), rim_idx=4, ball_xyz=ball, rim_distances=rim_dist
        )
        assert release == 1

        # The same arc out at the three-point line is not a shot at the rim.
        _, far = self._ball(z, rim_distance=25.0)
        assert (
            pairing.find_release(
                moments, np.array([1]), rim_idx=4, ball_xyz=ball, rim_distances=far
            )
            is None
        )

    def test_shooter_proximity_is_required_when_the_shooter_is_tracked(self):
        """Rather than fall back to a candidate known to be in the wrong place."""
        z = [4.0, 3.0, 3.0, 8.0, 14.0, 16.0, 12.0, 10.5]
        ball, rim_dist = self._ball(z, rim_distance=5.0)
        moments = _moments([1] * 8, np.arange(8)[::-1] * 0.2 + 250)

        far_away = np.full((8, 2), 40.0, dtype=np.float32)
        assert (
            pairing.find_release(
                moments, np.array([2]), 7, ball, rim_dist, shooter_xy=far_away
            )
            is None
        )

        at_the_ball = np.zeros((8, 2), dtype=np.float32)
        assert (
            pairing.find_release(
                moments, np.array([2]), 7, ball, rim_dist, shooter_xy=at_the_ball
            )
            == 2
        )

    def test_no_candidates(self):
        ball, rim_dist = self._ball([10.0, 10.0])
        moments = _moments([1, 1], [250.0, 249.8])
        assert pairing.find_release(moments, np.array([]), 1, ball, rim_dist) is None


class TestPairOnRealGame:
    """End-to-end against the sample game. Numbers are measured, see the module docstring."""

    @pytest.fixture(scope="class")
    @staticmethod
    def result(pbp_cache_dir, sample_game_id, sportvu_fixture_path):
        from rebounding.data import pbp, sportvu

        raw = pbp.to_frame(pbp.fetch(sample_game_id, cache_dir=pbp_cache_dir))
        shots = pbp.pair_shots_and_rebounds(raw, sample_game_id)
        tracking = sportvu.load(sportvu_fixture_path)
        return pairing.pair(tracking, shots, pbp.made_shots(raw))

    def test_produces_paired_shots_with_tracking_indices(self, result):
        paired, _ = result
        assert not paired.empty
        for column in ("ReleaseIdx", "RimIdx", "FlightTime", "Basket"):
            assert column in paired.columns

    def test_release_precedes_rim_arrival(self, result):
        paired, _ = result
        assert (paired["ReleaseIdx"] < paired["RimIdx"]).all()

    def test_flight_times_are_physically_plausible(self, result):
        paired, _ = result
        assert (paired["FlightTime"] > 0).all()
        assert (paired["FlightTime"] <= pairing.MAX_FLIGHT_SECONDS).all()

    def test_each_rim_arrival_is_used_at_most_once(self, result):
        paired, _ = result
        assert paired["RimIdx"].is_unique

    def test_basket_is_resolved_per_shot(self, result):
        paired, _ = result
        assert paired["Basket"].isin(["left", "right"]).all()

    def test_report_accounts_for_every_shot(self, result):
        paired, report = result
        assert report.n_paired == len(paired)
        assert report.n_paired + sum(report.drops.values()) == report.n_shots
        assert "paired" in str(report)


def test_pair_handles_an_empty_shot_list(sportvu_fixture_path):
    from rebounding.data import sportvu

    tracking = sportvu.load(sportvu_fixture_path)
    empty = pd.DataFrame(columns=["Period", "Clock", "ShootPlayerID", "BlockPlayerID"])
    paired, report = pairing.pair(tracking, empty)
    assert paired.empty
    assert report.n_shots == 0
    assert report.pair_rate == 0.0
