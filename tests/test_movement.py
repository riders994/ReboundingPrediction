"""The movement model, its scoring, and the feature bridge between them."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rebounding.constants import HOOP
from rebounding.data.features import N_PLAYERS, boxgen, rim_features
from rebounding.eval import movement as mv

torch = pytest.importorskip("torch", reason="the movement model is the only torch user")

from rebounding.models.movement import MOVE_SCALE, MovementConfig, MovementModel  # noqa: E402


def tiny(head: str, **kwargs) -> MovementConfig:
    """A config small enough to fit inside a test."""
    settings = dict(
        head=head, d_model=16, n_layers=2, n_heads=2, d_ff=32, d_latent=4,
        n_components=3, epochs=3, batch_size=64, patience=99,
    )
    return MovementConfig(**{**settings, **kwargs})


@pytest.fixture
def bimodal():
    """Players who either advance or retreat, never the average of the two.

    The whole argument for the density heads in one fixture: the conditional mean of
    this target is zero, and zero never happens.
    """
    rng = np.random.default_rng(0)
    n, n_features = 240, 6
    x = rng.normal(size=(n, N_PLAYERS, n_features)).astype(np.float32)
    forward = rng.random((n, N_PLAYERS)) < 0.5
    y = np.where(forward[..., None], [4.0, 0.0], [-4.0, 0.0])
    y = (y + rng.normal(scale=0.4, size=(n, N_PLAYERS, 2))).astype(np.float32)
    return x, y


# -- the network -----------------------------------------------------------------


@pytest.mark.parametrize("head", ["point", "mixture", "cvae"])
def test_every_head_fits_and_samples(bimodal, head):
    x, y = bimodal
    model = MovementModel(tiny(head)).fit(x, y, validation=(x[:60], y[:60]))

    assert model.predict(x[:7]).shape == (7, N_PLAYERS, 2)
    assert model.sample(x[:7], n=5).shape == (5, 7, N_PLAYERS, 2)
    assert all(math.isfinite(record["train_loss"]) for record in model.history)


@pytest.mark.parametrize("head", ["point", "mixture"])
def test_state_roundtrips(bimodal, head):
    x, y = bimodal
    model = MovementModel(tiny(head)).fit(x, y)
    restored = MovementModel.from_state(model.state())
    np.testing.assert_allclose(restored.predict(x[:9]), model.predict(x[:9]), atol=1e-5)


def test_the_encoder_is_permutation_equivariant(bimodal):
    """Shuffle the ten players and the ten predictions shuffle with them.

    This is the property that lets the app hand over players in whatever order a user
    placed them. It holds because there is no positional encoding, so it is worth a
    test rather than a comment -- adding one would break it silently.
    """
    x, y = bimodal
    model = MovementModel(tiny("point")).fit(x, y)

    order = np.array([7, 2, 9, 0, 4, 1, 8, 3, 6, 5])
    straight = model.predict(x[:16])
    shuffled = model.predict(x[:16][:, order])
    np.testing.assert_allclose(shuffled, straight[:, order], atol=1e-5)


def test_the_mixture_density_integrates_to_one(bimodal):
    """A grid quadrature over one player's predicted density, in feet."""
    x, y = bimodal
    model = MovementModel(tiny("mixture")).fit(x, y)

    step = 0.5
    axis = np.arange(-30.0, 30.0, step)
    grid = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)

    one = np.repeat(x[:1], len(grid), axis=0)
    target = np.zeros((len(grid), N_PLAYERS, 2), dtype=np.float32)
    target[:, 0] = grid
    mass = np.exp(model.log_prob(one, target)[:, 0]).sum() * step**2
    assert mass == pytest.approx(1.0, abs=0.02)


def test_log_prob_refuses_heads_without_a_density(bimodal):
    x, y = bimodal
    model = MovementModel(tiny("cvae")).fit(x, y)
    with pytest.raises(NotImplementedError, match="no closed-form density"):
        model.log_prob(x[:4], y[:4])


def test_the_scene_latent_couples_the_ten_players(bimodal):
    """Two draws from the VAE differ scene-wide, not player by player.

    A head that sampled each player independently would show no correlation between
    the players' deviations within a draw. The point head, by contrast, has no spread
    at all -- which is the same statement from the other side.
    """
    x, y = bimodal
    model = MovementModel(tiny("cvae", epochs=6)).fit(x, y)
    draws = model.sample(x[:64], n=32, seed=3)
    assert draws.std(axis=0).mean() > 0.05

    # The point head's "samples" are one prediction repeated, so its spread is zero
    # up to float32 summation noise -- which is the whole complaint against it.
    frozen = MovementModel(tiny("point")).fit(x, y)
    assert frozen.sample(x[:64], n=32, seed=3).std(axis=0).max() < 1e-5


def test_a_bad_target_shape_is_refused(bimodal):
    x, y = bimodal
    with pytest.raises(ValueError, match=r"target must be"):
        MovementModel(tiny("point")).fit(x, y[..., :1])


def test_move_scale_is_only_a_scale(bimodal):
    """Displacements come back in feet whatever the internal scaling is."""
    x, y = bimodal
    model = MovementModel(tiny("point")).fit(x, y)
    assert MOVE_SCALE > 1
    assert np.abs(model.predict(x[:32])).max() < 30.0


# -- the feature bridge ----------------------------------------------------------


def scene(rng, n=5):
    pre = np.concatenate(
        [rng.uniform(30, 45, size=(n, N_PLAYERS, 1)), rng.uniform(5, 45, size=(n, N_PLAYERS, 1))],
        axis=-1,
    )
    pos = pre + rng.normal(scale=5.0, size=pre.shape)
    shooter = np.zeros((n, N_PLAYERS))
    shooter[:, 0] = 1.0
    return pre, pos, shooter


def test_rim_features_agree_with_their_definitions():
    pre, pos, shooter = scene(np.random.default_rng(1))
    out = rim_features(pre, pos, shooter)

    np.testing.assert_allclose(out["move_dx"], pos[..., 0] - pre[..., 0])
    np.testing.assert_allclose(out["move_dist"], np.hypot(*(pos - pre).transpose(2, 0, 1)))
    np.testing.assert_allclose(out["pos_dist"], np.hypot(pos[..., 0] - HOOP[0], pos[..., 1] - HOOP[1]))
    closer = out["pos_dist"] < np.hypot(pre[..., 0] - HOOP[0], pre[..., 1] - HOOP[1])
    np.testing.assert_array_equal(out["closed_on_rim"], np.where(closer, 1.0, -1.0))
    # Every box-out count is one team's five players sharing five assignments.
    np.testing.assert_allclose(out["pos_box"].sum(axis=1), 10.0)


def test_rim_features_batches_the_same_way_it_loops():
    """The batched path is what serves predictions; the loop is what built the corpus."""
    pre, pos, shooter = scene(np.random.default_rng(2), n=4)
    batched = rim_features(pre, pos, shooter)
    for i in range(len(pre)):
        one = rim_features(pre[i : i + 1], pos[i : i + 1], shooter[i : i + 1])
        for name, values in one.items():
            np.testing.assert_allclose(values[0], batched[name][i], atol=1e-12)


def test_rim_features_checks_its_shapes():
    pre, pos, shooter = scene(np.random.default_rng(3))
    with pytest.raises(ValueError, match="matching"):
        rim_features(pre, pos[:, :9], shooter)


def test_pos_box_uses_the_same_counter_as_the_pipeline():
    pre, pos, shooter = scene(np.random.default_rng(4), n=3)
    out = rim_features(pre, pos, shooter)
    for i in range(len(pre)):
        np.testing.assert_allclose(out["pos_box"][i], boxgen(pos[i]))


# -- the metrics -----------------------------------------------------------------


def test_calibration_reads_a_calibrated_cloud_as_calibrated():
    rng = np.random.default_rng(5)
    # Truth and samples must be draws from the *same* conditional, each around a
    # shared mean the metric never sees. Centring the samples on the truth instead
    # would describe an oracle, and reads as wildly under-dispersed rather than
    # calibrated -- the centroid lands on the truth and every sample is further out.
    mean = rng.normal(scale=10.0, size=(400, N_PLAYERS, 2))
    truth = mean + rng.normal(scale=3.0, size=mean.shape)
    samples = mean[None] + rng.normal(scale=3.0, size=(200, *mean.shape))
    report = mv.radial_calibration(samples, truth)
    assert report["mean_rank"] == pytest.approx(0.5, abs=0.05)


def test_calibration_reads_a_point_predictor_as_overconfident():
    rng = np.random.default_rng(6)
    truth = rng.normal(scale=3.0, size=(200, N_PLAYERS, 2))
    samples = np.zeros((10, *truth.shape))  # every draw identical
    report = mv.radial_calibration(samples, truth)
    assert report["coverage_50"] == 0.0
    assert report["coverage_90"] == 0.0
    assert report["spread_ft"] == 0.0


def test_min_ade_rewards_covering_the_truth():
    rng = np.random.default_rng(7)
    truth = rng.normal(size=(50, N_PLAYERS, 2))
    lucky = np.concatenate([truth[None], rng.normal(size=(9, 50, N_PLAYERS, 2)) * 10], axis=0)
    assert mv.min_ade(lucky, truth) == pytest.approx(0.0, abs=1e-6)
    # Scene-level is never easier: it has to be the same draw for all ten.
    assert mv.min_ade(lucky, truth, scene=True) >= mv.min_ade(lucky, truth)


def test_crash_rates_split_offence_from_defence():
    release = np.tile(np.array([20.0, 25.0]), (30, N_PLAYERS, 1))
    positions = release.copy()
    is_offense = np.tile(np.r_[np.ones(5), np.zeros(5)], (30, 1))
    positions[:, :5, 0] += 5.0  # offence moves toward the rim at x=41.75
    positions[:, 5:, 0] -= 5.0  # defence moves away

    rates = mv.crash_rates(positions, release, is_offense)
    assert rates["crash_offense"] == 1.0
    assert rates["crash_defense"] == 0.0


def test_plausibility_counts_contacts_and_impossible_speeds():
    positions = np.zeros((4, N_PLAYERS, 2))
    positions[:, :, 0] = np.arange(N_PLAYERS) * 10.0
    positions[:, 1, 0] = 0.5  # players 0 and 1 on top of each other
    release = np.zeros_like(positions)
    flight = np.full((4, 1), 0.1)  # far too short for the distances above

    report = mv.plausibility(positions, flight, release)
    assert report["contacts_per_scene"] == pytest.approx(1.0)
    assert report["impossible_speed_rate"] > 0.5
