"""Tests for the serving entry point.

The module exists to make the app and the pipeline agree, so the central test is
**parity**: take real feature rows the pipeline built from the committed sample game,
hand the same positions back to `serve.feature_frame`, and require the numbers to come
out identical. Everything §3 of the handoff brief lists -- the hoop, the angle
convention, the box-out count, the slot order -- is covered by that one assertion, and
covered against the pipeline's own output rather than against a restatement of it.

The other half is the reordering. Players arrive in whatever order the UI holds them
and are sorted into canonical slots to be scored, so the probabilities have to be put
back. A permutation test pins that: the same ten players in a different order must get
the same ten probabilities.
"""

import numpy as np
import pandas as pd
import pytest

from rebounding.constants import DEFAULT_POSITION
from rebounding.data import features as features_module
from rebounding.data.court import LEFT, RIGHT, unfold
from rebounding.data.derived import ShotPriors
from rebounding.data.features import SERVED_FEATURES
from rebounding.serve import (
    PlacementError,
    Player,
    ShotPrediction,
    feature_frame,
    predict,
)

N_PLAYERS = 10

# The columns serve.py builds itself; the rest come from the fitted priors.
BASE_COLUMNS = [
    "pre_x", "pre_y", "pre_dist", "pre_angle", "pre_cos_shooter", "pre_box",
    "is_offense", "is_shooter", "role",
]


@pytest.fixture(scope="module")
def rows(pbp_cache_dir, sample_game_id, sportvu_fixture_path):
    """Real pipeline feature rows for the committed sample game."""
    from rebounding.data import pairing, pbp, sportvu

    raw = pbp.to_frame(pbp.fetch(sample_game_id, cache_dir=pbp_cache_dir))
    shots = pbp.pair_shots_and_rebounds(raw, sample_game_id)
    tracking = sportvu.load(sportvu_fixture_path)
    paired, _ = pairing.pair(tracking, shots, pbp.made_shots(raw))
    return features_module.build(tracking, paired)


@pytest.fixture(scope="module")
def priors(rows):
    return ShotPriors().fit(rows)


def _players_from(shot: pd.DataFrame) -> list[Player]:
    """Reconstruct the placed dots a user would have made for one pipeline shot."""
    return [
        Player(
            x=row.pre_x,
            y=row.pre_y,
            is_offense=bool(row.is_offense),
            is_shooter=bool(row.is_shooter),
            position=row.role,
        )
        for row in shot.itertuples()
    ]


class StubArtifact:
    """An artifact whose scores identify the slot, so reordering is observable."""

    def __init__(self, priors, features=None):
        self.priors = priors
        self.features = list(features or SERVED_FEATURES)
        self.regime = "served"

    def predict_proba(self, x):
        n_shots, n_slots, _ = x.shape
        weights = np.arange(1, n_slots + 1, dtype=float)
        return np.tile(weights / weights.sum(), (n_shots, 1))


# --------------------------------------------------------------------------- #
# Parity with the pipeline
# --------------------------------------------------------------------------- #


def test_reproduces_the_pipeline_features_exactly(rows):
    """The whole point of the module: no train/serve skew, to floating point."""
    checked = 0
    for _, shot in rows.groupby("ShotID", sort=False):
        shot = shot.sort_values("Slot")
        served = feature_frame(_players_from(shot))
        for column in BASE_COLUMNS:
            np.testing.assert_allclose(
                served[column].to_numpy(float), shot[column].to_numpy(float),
                atol=1e-9, err_msg=f"{column} diverged from the pipeline",
            )
        checked += 1
    assert checked > 0


def test_derived_features_match_the_pipeline(rows, priors):
    """`ShotPriors.transform` gets fed zero velocities; served columns must not care."""
    shot = rows[rows["ShotID"] == rows["ShotID"].iloc[0]].sort_values("Slot")
    served = priors.transform(feature_frame(_players_from(shot)))
    truth = priors.transform(shot)
    for column in SERVED_FEATURES:
        np.testing.assert_allclose(
            served[column].to_numpy(float), truth[column].to_numpy(float),
            atol=1e-9, err_msg=f"{column} depends on a velocity the app cannot supply",
        )


def test_slot_order_is_offense_first_then_nearest_rim(rows):
    shot = rows[rows["ShotID"] == rows["ShotID"].iloc[0]].sort_values("Slot")
    served = feature_frame(_players_from(shot))
    assert list(served["is_offense"]) == [1.0] * 5 + [0.0] * 5
    for block in (served.iloc[:5], served.iloc[5:]):
        assert list(block["pre_dist"]) == sorted(block["pre_dist"])


# --------------------------------------------------------------------------- #
# Order in, order out
# --------------------------------------------------------------------------- #


def test_probabilities_come_back_in_the_callers_order(rows, priors):
    """Slot-order output would mis-assign every dot while looking plausible."""
    shot = rows[rows["ShotID"] == rows["ShotID"].iloc[0]].sort_values("Slot")
    players = _players_from(shot)
    artifact = StubArtifact(priors)

    prediction = predict(artifact, players)
    expected = np.arange(1, N_PLAYERS + 1) / np.arange(1, N_PLAYERS + 1).sum()
    # Each input player's probability is the one belonging to the slot he was sorted
    # into, not the one sitting at his own index.
    np.testing.assert_allclose(prediction.probabilities, expected[prediction.slots])


def test_shuffling_the_input_does_not_change_any_players_probability(rows, priors):
    shot = rows[rows["ShotID"] == rows["ShotID"].iloc[0]].sort_values("Slot")
    players = _players_from(shot)
    artifact = StubArtifact(priors)

    straight = predict(artifact, players)
    permutation = np.array([7, 2, 9, 0, 4, 1, 8, 3, 6, 5])
    shuffled = predict(artifact, [players[i] for i in permutation])

    np.testing.assert_allclose(shuffled.probabilities, straight.probabilities[permutation])
    assert players[straight.most_likely()] is players[permutation[shuffled.most_likely()]]


def test_ranked_is_sorted_and_indexes_the_input(rows, priors):
    shot = rows[rows["ShotID"] == rows["ShotID"].iloc[0]].sort_values("Slot")
    prediction = predict(StubArtifact(priors), _players_from(shot))

    ranked = prediction.ranked()
    assert [i for i, _ in ranked] != sorted(i for i, _ in ranked) or len(set(ranked)) == 1
    assert [p for _, p in ranked] == sorted((p for _, p in ranked), reverse=True)
    assert ranked[0][0] == prediction.most_likely()
    assert sorted(i for i, _ in ranked) == list(range(N_PLAYERS))


def test_probabilities_sum_to_one_without_renormalising(rows, priors):
    shot = rows[rows["ShotID"] == rows["ShotID"].iloc[0]].sort_values("Slot")
    prediction = predict(StubArtifact(priors), _players_from(shot))
    assert prediction.probabilities.sum() == pytest.approx(1.0)
    assert isinstance(prediction, ShotPrediction)


# --------------------------------------------------------------------------- #
# The coordinate frame
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("basket", [LEFT, RIGHT])
def test_full_court_input_folds_to_the_same_thing(rows, basket):
    """Unfold the pipeline's folded positions, hand them back, and get them again."""
    shot = rows[rows["ShotID"] == rows["ShotID"].iloc[0]].sort_values("Slot")
    folded = feature_frame(_players_from(shot))

    full_court = unfold(shot[["pre_x", "pre_y"]].to_numpy(float), basket)
    players = [
        Player(x=xy[0], y=xy[1], is_offense=bool(r.is_offense),
               is_shooter=bool(r.is_shooter), position=r.role)
        for xy, r in zip(full_court, shot.itertuples(), strict=True)
    ]
    np.testing.assert_allclose(
        feature_frame(players, basket=basket)["pre_x"].to_numpy(), folded["pre_x"].to_numpy(),
        atol=1e-9,
    )


def test_rejects_unfolded_coordinates_in_the_far_half(rows):
    shot = rows[rows["ShotID"] == rows["ShotID"].iloc[0]].sort_values("Slot")
    players = [
        Player(x=p.x + 47.0, y=p.y, is_offense=p.is_offense, is_shooter=p.is_shooter)
        for p in _players_from(shot)
    ]
    with pytest.raises(PlacementError, match="x out of range"):
        feature_frame(players)


def test_accepts_the_backcourt_and_behind_the_baseline():
    """The corpus contains both; a tight box would reject real placements."""
    players = [
        Player(x=x, y=25.0, is_offense=i < 5, is_shooter=i == 0)
        for i, x in enumerate([-40.0, 52.0, 10.0, 20.0, 30.0, 5.0, 15.0, 25.0, 35.0, 45.0])
    ]
    assert len(feature_frame(players)) == N_PLAYERS


def test_rejects_y_off_the_floor():
    players = [
        Player(x=20.0, y=90.0 if i == 0 else 25.0, is_offense=i < 5, is_shooter=i == 0)
        for i in range(N_PLAYERS)
    ]
    with pytest.raises(PlacementError, match="y out of range"):
        feature_frame(players)


# --------------------------------------------------------------------------- #
# Refusing to serve what it cannot build
# --------------------------------------------------------------------------- #


def test_refuses_an_artifact_that_wants_velocity(rows, priors):
    """Zeros here are train/serve skew, and worse than the honest no-velocity model."""
    shot = rows[rows["ShotID"] == rows["ShotID"].iloc[0]].sort_values("Slot")
    artifact = StubArtifact(priors, features=[*SERVED_FEATURES, "pre_speed"])
    with pytest.raises(PlacementError, match="cannot be built from static"):
        predict(artifact, _players_from(shot))


@pytest.mark.parametrize("velocity_feature", ["pre_vx", "v_radial", "ext_dist"])
def test_every_velocity_derived_feature_is_refused(rows, priors, velocity_feature):
    shot = rows[rows["ShotID"] == rows["ShotID"].iloc[0]].sort_values("Slot")
    artifact = StubArtifact(priors, features=[*SERVED_FEATURES, velocity_feature])
    with pytest.raises(PlacementError, match="cannot be built from static"):
        predict(artifact, _players_from(shot))


# --------------------------------------------------------------------------- #
# Placement validation
# --------------------------------------------------------------------------- #


def _ten(**overrides) -> list[Player]:
    players = [
        Player(x=10.0 + i, y=25.0, is_offense=i < 5, is_shooter=i == 0) for i in range(N_PLAYERS)
    ]
    for index, player in overrides.items():
        players[int(index)] = player
    return players


def test_rejects_the_wrong_number_of_players():
    with pytest.raises(PlacementError, match="expected 10 players"):
        feature_frame(_ten()[:9])


def test_rejects_an_uneven_team_split():
    players = _ten()
    players[9] = Player(x=19.0, y=25.0, is_offense=True)
    with pytest.raises(PlacementError, match="expected 5 offensive players"):
        feature_frame(players)


def test_rejects_no_shooter():
    players = _ten()
    players[0] = Player(x=10.0, y=25.0, is_offense=True, is_shooter=False)
    with pytest.raises(PlacementError, match="exactly 1 shooter"):
        feature_frame(players)


def test_rejects_two_shooters():
    players = _ten()
    players[1] = Player(x=11.0, y=25.0, is_offense=True, is_shooter=True)
    with pytest.raises(PlacementError, match="exactly 1 shooter"):
        feature_frame(players)


def test_rejects_a_defensive_shooter():
    players = _ten()
    players[0] = Player(x=10.0, y=25.0, is_offense=True, is_shooter=False)
    players[7] = Player(x=17.0, y=25.0, is_offense=False, is_shooter=True)
    with pytest.raises(PlacementError, match="shooter must be on the offensive team"):
        feature_frame(players)


# --------------------------------------------------------------------------- #
# The JSON the front end posts
# --------------------------------------------------------------------------- #


def test_accepts_plain_dicts():
    payload = [
        {"x": 10.0 + i, "y": 25.0, "is_offense": i < 5, "is_shooter": i == 0}
        for i in range(N_PLAYERS)
    ]
    assert len(feature_frame(payload)) == N_PLAYERS


def test_names_the_missing_key():
    payload = [{"x": 10.0, "y": 25.0} for _ in range(N_PLAYERS)]
    with pytest.raises(PlacementError, match="missing required key 'is_offense'"):
        feature_frame(payload)


@pytest.mark.parametrize(
    ("position", "expected"),
    [("G", 1.0), ("c", 5.0), ("F-C", 11.0 / 3), (" C ", 5.0), (5.0, 5.0),
     (None, DEFAULT_POSITION), ("point guard", DEFAULT_POSITION)],
)
def test_role_encoding(position, expected):
    assert Player(x=0.0, y=0.0, is_offense=True, position=position).role == pytest.approx(expected)


def test_player_id_is_carried_through():
    payload = [
        {"x": 10.0 + i, "y": 25.0, "is_offense": i < 5, "is_shooter": i == 0, "player_id": f"p{i}"}
        for i in range(N_PLAYERS)
    ]
    frame = feature_frame(payload)
    assert set(frame["PlayerID"]) == {f"p{i}" for i in range(N_PLAYERS)}


# --------------------------------------------------------------------------- #
# The movement model's two roles: the animation, and the extra ten columns
# --------------------------------------------------------------------------- #


class StubMovement:
    """A movement model that walks everyone one foot toward half court.

    Deterministic on purpose, and with a spread that comes from the sample index
    rather than from a random draw, so both the ordering and the sampling can be
    asserted without a trained network in the test.
    """

    def __init__(self, priors, features=None):
        self.priors = priors
        self.features = list(features or SERVED_FEATURES)

    def predict_positions(self, x, release_xy):
        step = np.zeros_like(release_xy)
        step[..., 0] = -1.0
        return release_xy + step

    def sample_positions(self, x, release_xy, n=20, seed=None):
        offsets = np.arange(n, dtype=float).reshape(n, 1, 1, 1)
        return self.predict_positions(x, release_xy)[None] + offsets

    def rim_block(self, x, release_xy, is_shooter, positions=None):
        from rebounding.data.features import MOVEMENT_SUPPLIED, rim_features

        if positions is None:
            positions = self.predict_positions(x, release_xy)
        columns = rim_features(release_xy, positions, is_shooter)
        return np.stack([columns[c] for c in MOVEMENT_SUPPLIED], axis=-1).astype(np.float32)


def test_animate_returns_scenes_in_the_callers_order(rows, priors):
    from rebounding.serve import animate

    shot = next(iter(rows.groupby("ShotID", sort=False)))[1].sort_values("Slot")
    players = _players_from(shot)
    # Hand them over in an order the pipeline would never produce.
    shuffled = [players[i] for i in (9, 4, 1, 7, 0, 3, 8, 2, 6, 5)]

    result = animate(StubMovement(priors), shuffled, n=4)
    assert result.scenes.shape == (4, 10, 2)
    for index, player in enumerate(shuffled):
        assert result.mean[index, 0] == pytest.approx(player.x - 1.0)
        assert result.mean[index, 1] == pytest.approx(player.y)


def test_animate_samples_differ_from_each_other(rows, priors):
    from rebounding.serve import animate

    shot = next(iter(rows.groupby("ShotID", sort=False)))[1].sort_values("Slot")
    result = animate(StubMovement(priors), _players_from(shot), n=3)
    assert not np.allclose(result.scene(0), result.scene(1))


def test_a_movement_hungry_artifact_is_refused_without_one(rows, priors):
    """Rim-time columns cannot be read off a shot that has not landed."""
    from rebounding.data.features import SERVED_PLUS_MOVEMENT

    shot = next(iter(rows.groupby("ShotID", sort=False)))[1].sort_values("Slot")
    artifact = StubArtifact(priors, features=SERVED_PLUS_MOVEMENT)
    with pytest.raises(PlacementError, match="rim-time positions"):
        predict(artifact, _players_from(shot))


def test_the_movement_columns_reach_the_rebounder(rows, priors):
    from rebounding.data.features import SERVED_PLUS_MOVEMENT

    shot = next(iter(rows.groupby("ShotID", sort=False)))[1].sort_values("Slot")
    artifact = StubArtifact(priors, features=SERVED_PLUS_MOVEMENT)
    prediction = predict(artifact, _players_from(shot), movement=StubMovement(priors))

    assert prediction.probabilities.shape == (10,)
    # The stub walks every player a foot toward half court, so `pos_x` has to be
    # `pre_x - 1` in the frame the trees were handed -- not the corpus's real value.
    np.testing.assert_allclose(
        prediction.features["pos_x"].to_numpy(float),
        prediction.features["pre_x"].to_numpy(float) - 1.0,
    )
    np.testing.assert_allclose(prediction.features["move_dx"].to_numpy(float), -1.0)


class TeleportingMovement(StubMovement):
    """A movement model whose draws are physically impossible, on demand.

    The real failure is a tail: the decoder is Gaussian, so a small fraction of draws
    land somewhere no player could reach in a flight. Reproducing that with a stub
    means choosing when it happens rather than sampling until it does.
    """

    def __init__(self, priors, features=None, bad_draws=2, distance=400.0):
        super().__init__(priors, features)
        self.bad_draws, self.distance = bad_draws, distance
        self.calls = 0

    def sample_positions(self, x, release_xy, n=20, seed=None):
        self.calls += 1
        scenes = super().sample_positions(x, release_xy, n=n, seed=seed)
        if self.calls == 1:
            scenes[: self.bad_draws, ..., 0] += self.distance
        return scenes


def test_impossible_scenes_are_redrawn_rather_than_drawn(rows, priors):
    """~1 scene in 100 from the real model needs a player to outrun the corpus."""
    from rebounding.constants import MAX_SPEED_FPS
    from rebounding.serve import animate

    shot = next(iter(rows.groupby("ShotID", sort=False)))[1].sort_values("Slot")
    players = _players_from(shot)
    movement = TeleportingMovement(priors, bad_draws=2)

    result = animate(movement, players, n=6, seed=1)

    flight = float(result.features["flight_hat"].iloc[0])
    release = np.array([[p.x, p.y] for p in players])
    speed = np.linalg.norm(result.scenes - release, axis=-1) / flight
    assert speed.max() <= MAX_SPEED_FPS
    assert result.redrawn == 2
    assert result.clamped == 0
    assert movement.calls > 1


def test_a_scene_that_is_already_possible_is_left_alone(rows, priors):
    """The filter must not quietly reshape the 99% of draws that are fine."""
    from rebounding.serve import animate

    shot = next(iter(rows.groupby("ShotID", sort=False)))[1].sort_values("Slot")
    players = _players_from(shot)

    plain = animate(StubMovement(priors), players, n=5, seed=1)
    assert plain.redrawn == 0 and plain.clamped == 0
    np.testing.assert_allclose(
        plain.scenes, animate(StubMovement(priors), players, n=5, seed=1).scenes
    )


def test_a_scene_that_stays_impossible_is_clamped_not_looped(priors):
    """Rejection cannot be unbounded, so the last resort is to cut the distance."""
    from rebounding.constants import MAX_SPEED_FPS
    from rebounding.serve import _reject_impossible

    release = np.tile(np.array([20.0, 25.0]), (10, 1))
    scenes = np.tile(release, (4, 1, 1))
    scenes[..., 0] += 500.0  # every player, every scene, hopelessly far

    filtered, redrawn, clamped = _reject_impossible(
        lambda count: scenes[:count].copy(), scenes.copy(), release, flight=1.9
    )

    speed = np.linalg.norm(filtered - release, axis=-1) / 1.9
    assert speed.max() == pytest.approx(MAX_SPEED_FPS)
    assert clamped == 40 and redrawn == 12
    # The clamp keeps each player's heading and only takes back the distance.
    direction = (filtered - release) / np.linalg.norm(filtered - release, axis=-1)[..., None]
    assert direction[..., 0] == pytest.approx(1.0)
