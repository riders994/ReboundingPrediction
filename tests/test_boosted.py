"""Tests for the boosted grouped-softmax model.

The custom objective is the part worth testing directly: LightGBM hands it a flat
array of every training row and trusts whatever gradient comes back, so an error in
the reshape or the sign produces a model that trains quietly and predicts badly. The
gradient is checked against a finite difference of the loss it claims to be
differentiating.
"""

import numpy as np
import pytest

from rebounding.eval.metrics import evaluate
from rebounding.models.boosted import BoostedSoftmax, grouped_softmax_objective

pytest.importorskip("lightgbm")

N_PLAYERS = 10


def negative_log_likelihood(raw, labels):
    """The loss `grouped_softmax_objective` is the derivative of."""
    scores = raw.reshape(-1, N_PLAYERS)
    shifted = scores - scores.max(axis=1, keepdims=True)
    log_p = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return -(labels.reshape(-1, N_PLAYERS) * log_p).sum()


@pytest.fixture(scope="module")
def separable():
    """Shots where the nearest player rebounds, plus a column of pure noise."""
    rng = np.random.default_rng(0)
    n_shots = 1200
    distance = rng.uniform(1, 30, size=(n_shots, N_PLAYERS))
    noise = rng.normal(size=(n_shots, N_PLAYERS))
    x = np.stack([distance, noise], axis=2).astype(np.float32)
    labels = np.zeros((n_shots, N_PLAYERS), dtype=np.int8)
    labels[np.arange(n_shots), distance.argmin(axis=1)] = 1
    return x, labels


# --------------------------------------------------------------------------- #
# The objective
# --------------------------------------------------------------------------- #


def test_gradient_matches_a_finite_difference():
    rng = np.random.default_rng(1)
    raw = rng.normal(size=3 * N_PLAYERS)
    labels = np.zeros((3, N_PLAYERS))
    labels[np.arange(3), [2, 7, 0]] = 1.0
    flat = labels.reshape(-1)

    gradient, _ = grouped_softmax_objective(flat, raw)

    step = 1e-6
    numeric = np.empty_like(raw)
    for i in range(len(raw)):
        up, down = raw.copy(), raw.copy()
        up[i] += step
        down[i] -= step
        numeric[i] = (negative_log_likelihood(up, flat)
                      - negative_log_likelihood(down, flat)) / (2 * step)

    assert np.allclose(gradient, numeric, atol=1e-6)


def test_gradient_sums_to_zero_within_a_shot():
    """The softmax is over the group, so shifting all ten scores changes nothing."""
    rng = np.random.default_rng(2)
    raw = rng.normal(size=5 * N_PLAYERS)
    labels = np.zeros((5, N_PLAYERS))
    labels[np.arange(5), rng.integers(0, N_PLAYERS, 5)] = 1.0

    gradient, _ = grouped_softmax_objective(labels.reshape(-1), raw)
    assert np.allclose(gradient.reshape(5, N_PLAYERS).sum(axis=1), 0.0, atol=1e-12)


def test_hessian_is_bounded_away_from_zero():
    """A saturated score would otherwise give LightGBM an unbounded leaf value."""
    raw = np.full(N_PLAYERS, -50.0)
    raw[3] = 50.0
    labels = np.zeros(N_PLAYERS)
    labels[3] = 1.0
    _, hessian = grouped_softmax_objective(labels, raw)
    assert (hessian > 0).all()
    assert np.isfinite(hessian).all()


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #


def test_probabilities_are_normalised_within_each_shot(separable):
    x, y = separable
    probabilities = BoostedSoftmax(n_estimators=40).fit(x, y).predict_proba(x)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert (probabilities >= 0).all()


def test_learns_a_recoverable_signal(separable):
    x, y = separable
    model = BoostedSoftmax(n_estimators=200, learning_rate=0.1, num_leaves=31).fit(x, y)
    scores = evaluate(model.predict_proba(x), y)
    assert scores.top1 > 0.9


def test_importance_finds_the_feature_that_matters(separable):
    x, y = separable
    model = BoostedSoftmax(n_estimators=120, learning_rate=0.1).fit(x, y)
    importances = model.importances(["distance", "noise"])
    assert next(iter(importances)) == "distance"
    assert importances["distance"] > importances["noise"]


def test_rejects_a_group_that_is_not_ten_players():
    model = BoostedSoftmax(n_estimators=5)
    with pytest.raises(ValueError, match="players per shot"):
        model.fit(np.zeros((4, 6, 2)), np.zeros((4, 6), dtype=np.int8))


def test_refuses_to_predict_before_being_fitted():
    with pytest.raises(RuntimeError, match="not fitted"):
        BoostedSoftmax().predict_proba(np.zeros((2, N_PLAYERS, 3)))
