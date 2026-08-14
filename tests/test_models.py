"""Tests for the grouped-softmax model, the baselines, and the metrics.

The synthetic fixtures here have a known answer, which is the point: on data where
the right coefficient sign is arithmetic rather than opinion, a model that cannot
recover it is broken regardless of what it scores on the real frame.
"""

import numpy as np
import pytest

from rebounding.eval.metrics import evaluate
from rebounding.models.baselines import NearestPlayer, SlotPrior
from rebounding.models.conditional_logit import ConditionalLogit

N_PLAYERS = 10


@pytest.fixture(scope="module")
def distance_only():
    """Shots where the nearest player always rebounds, with one distance feature."""
    rng = np.random.default_rng(0)
    distances = rng.uniform(1, 30, size=(800, N_PLAYERS, 1)).astype(np.float32)
    winners = distances[:, :, 0].argmin(axis=1)
    labels = np.zeros((800, N_PLAYERS), dtype=np.int8)
    labels[np.arange(800), winners] = 1
    return distances, labels


# --------------------------------------------------------------------------- #
# Conditional logit
# --------------------------------------------------------------------------- #


def test_probabilities_are_normalised_within_each_shot(distance_only):
    x, y = distance_only
    probabilities = ConditionalLogit().fit(x, y).predict_proba(x)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert (probabilities >= 0).all()


def test_recovers_the_sign_of_a_known_effect(distance_only):
    """Closer means likelier, so the distance coefficient must be negative."""
    x, y = distance_only
    model = ConditionalLogit(l2=0.01).fit(x, y)
    assert model.converged_
    assert model.coefficients(["dist"])["dist"] < 0


def test_learns_a_recoverable_signal(distance_only):
    x, y = distance_only
    model = ConditionalLogit(l2=0.01).fit(x, y)
    scores = evaluate(model.predict_proba(x), y)
    # The label is a deterministic function of the one feature, so this should be
    # nearly perfect; anything near chance (10%) means the fit is not working.
    assert scores.top1 > 0.95


def test_slot_bias_alone_reproduces_the_empirical_rates():
    """With no usable features the model must fall back to the base rate per slot."""
    rng = np.random.default_rng(1)
    # One constant feature: it cannot discriminate, because a value identical across
    # the ten players cancels in a softmax over those ten.
    x = np.ones((2000, N_PLAYERS, 1), dtype=np.float32)
    winners = rng.choice(N_PLAYERS, size=2000, p=[0.3, 0.2, 0.15, 0.1, 0.08, 0.06, 0.05, 0.03, 0.02, 0.01])
    y = np.zeros((2000, N_PLAYERS), dtype=np.int8)
    y[np.arange(2000), winners] = 1

    model = ConditionalLogit(l2=0.0, fit_slot_bias=True).fit(x, y)
    predicted = model.predict_proba(x)[0]
    assert np.allclose(predicted, y.mean(axis=0), atol=0.02)


def test_without_slot_bias_a_constant_feature_predicts_uniformly():
    x = np.ones((500, N_PLAYERS, 1), dtype=np.float32)
    y = np.zeros((500, N_PLAYERS), dtype=np.int8)
    y[:, 0] = 1
    model = ConditionalLogit(fit_slot_bias=False).fit(x, y)
    assert np.allclose(model.predict_proba(x), 1.0 / N_PLAYERS)


def test_rejects_wrong_player_count():
    x = np.zeros((5, 7, 3), dtype=np.float32)
    y = np.zeros((5, 7), dtype=np.int8)
    with pytest.raises(ValueError, match="expected 10 players"):
        ConditionalLogit().fit(x, y)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        ConditionalLogit().predict_proba(np.zeros((2, N_PLAYERS, 1), dtype=np.float32))


def test_constant_columns_do_not_produce_nans():
    """A zero-variance column would divide by zero when standardising."""
    rng = np.random.default_rng(2)
    x = np.concatenate(
        [rng.uniform(1, 30, size=(200, N_PLAYERS, 1)), np.ones((200, N_PLAYERS, 1))], axis=2
    ).astype(np.float32)
    y = np.zeros((200, N_PLAYERS), dtype=np.int8)
    y[np.arange(200), x[:, :, 0].argmin(axis=1)] = 1

    probabilities = ConditionalLogit().fit(x, y).predict_proba(x)
    assert np.isfinite(probabilities).all()


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #


def test_slot_prior_learns_the_training_rates(distance_only):
    x, y = distance_only
    prior = SlotPrior().fit(x, y)
    probabilities = prior.predict_proba(x)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    # Identical for every shot: it ignores the features entirely.
    assert np.allclose(probabilities[0], probabilities[-1])


def test_nearest_player_picks_the_minimum_distance(distance_only):
    x, y = distance_only
    model = NearestPlayer(["dist"], "dist")
    predicted = model.predict_proba(x).argmax(axis=1)
    assert (predicted == x[:, :, 0].argmin(axis=1)).all()


def test_nearest_player_rejects_an_absent_column():
    with pytest.raises(ValueError, match="not among the supplied features"):
        NearestPlayer(["pre_x", "pre_y"], "pos_dist")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def test_top1_and_top3_on_a_known_ranking():
    probabilities = np.array(
        [
            [0.5, 0.2, 0.1, 0.05, 0.05, 0.03, 0.03, 0.02, 0.01, 0.01],  # truth rank 1
            [0.1, 0.5, 0.2, 0.05, 0.05, 0.03, 0.03, 0.02, 0.01, 0.01],  # truth rank 2
            # Truth is tied for last with one other, so pessimistically rank 10.
            [0.1, 0.2, 0.5, 0.05, 0.05, 0.03, 0.03, 0.02, 0.01, 0.01],  # truth rank 10
        ]
    )
    labels = np.zeros((3, N_PLAYERS), dtype=np.int8)
    labels[0, 0] = 1
    labels[1, 2] = 1
    labels[2, 9] = 1

    scores = evaluate(probabilities, labels)
    assert scores.top1 == pytest.approx(1 / 3)
    assert scores.top3 == pytest.approx(2 / 3)
    assert scores.mrr == pytest.approx((1 / 1 + 1 / 2 + 1 / 10) / 3)


def test_ties_are_broken_pessimistically():
    """A tie is not a win: the model still has to commit to one player."""
    probabilities = np.full((1, N_PLAYERS), 0.1)
    labels = np.zeros((1, N_PLAYERS), dtype=np.int8)
    labels[0, 4] = 1
    scores = evaluate(probabilities, labels)
    assert scores.top1 == 0.0
    assert scores.top3 == 0.0


def test_rejects_shots_without_exactly_one_rebounder():
    probabilities = np.full((2, N_PLAYERS), 0.1)
    labels = np.zeros((2, N_PLAYERS), dtype=np.int8)
    labels[0, 0] = 1  # second shot has no rebounder
    with pytest.raises(ValueError, match="exactly one labelled rebounder"):
        evaluate(probabilities, labels)


def test_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        evaluate(np.zeros((3, N_PLAYERS)), np.zeros((4, N_PLAYERS), dtype=np.int8))
