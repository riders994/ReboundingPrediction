"""Gradient boosting under the grouped-softmax loss.

This is :mod:`rebounding.models.conditional_logit` with the linear score replaced by
a tree ensemble. The loss is unchanged -- the per-shot cross entropy over the ten
players -- which is the point. Three ways of getting a tree ensemble to answer
"which of these ten" were measured on the validation split, all on the same
release-time features:

===============================================  =======
model                                              top-1
===============================================  =======
random forest, per-row binary, ranked after       27.6%
LightGBM ``lambdarank``, groups of ten            28.0%
LightGBM, grouped softmax (this)                  28.5%
===============================================  =======

The ordering is what the conditional logit's docstring predicts. A per-row binary
fit optimises "is this player the rebounder" independently for each of the ten and
then has its probabilities renormalised afterwards, so nothing in training ever sees
the constraint that exactly one of them is. ``lambdarank`` optimises a ranking
surrogate rather than the likelihood. The grouped softmax optimises the quantity
being measured.

The gradient and Hessian are the standard softmax ones, ``p - y`` and ``p(1 - p)``,
computed per shot. LightGBM hands the objective every row of the training set in one
flat array; the rows arrive in the order they were supplied, so the reshape to
``(n_shots, 10)`` recovers the groups as long as the caller passes the tensor from
:func:`rebounding.data.features.to_tensor`, which is already shot-major.

Depth is the parameter that matters here and it wants to be small: 127 leaves scored
a point worse than 63 on every learning rate tried, which is what 27k training shots
buys. The corpus cannot grow -- see ``data/MANIFEST.md`` -- so that ceiling is fixed.
"""

from __future__ import annotations

import numpy as np

N_PLAYERS = 10


def _grouped_softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def grouped_softmax_objective(labels: np.ndarray, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """LightGBM custom objective: cross entropy over each shot's ten players.

    Module level rather than a method so the estimator stays picklable -- sklearn
    deep-copies its parameters, and a bound method drags the whole object along.
    """
    n_shots = len(labels) // N_PLAYERS
    probabilities = _grouped_softmax(raw.reshape(n_shots, N_PLAYERS))
    gradient = (probabilities - labels.reshape(n_shots, N_PLAYERS)).reshape(-1)
    hessian = (probabilities * (1.0 - probabilities)).reshape(-1)
    # A player the model is certain about contributes a vanishing Hessian, which
    # LightGBM would turn into an unbounded leaf value.
    return gradient, np.maximum(hessian, 1e-6)


class BoostedSoftmax:
    """Boosted scorer over the ten players of a shot, normalised within the shot.

    Same interface as the rest of the ladder: ``x`` is ``(n_shots, 10, n_features)``
    and ``y`` is ``(n_shots, 10)`` one-hot.
    """

    def __init__(
        self,
        n_estimators: int = 800,
        learning_rate: float = 0.03,
        num_leaves: int = 63,
        min_child_samples: int = 80,
        reg_lambda: float = 1.0,
        n_jobs: int = -1,
        random_state: int = 0,
    ) -> None:
        self.params = dict(
            objective=grouped_softmax_objective,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            reg_lambda=reg_lambda,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=-1,
        )
        self.model = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> BoostedSoftmax:
        import lightgbm

        n_shots, n_slots, n_features = x.shape
        if n_slots != N_PLAYERS:
            raise ValueError(f"expected {N_PLAYERS} players per shot, got {n_slots}")

        self.model = lightgbm.LGBMRegressor(**self.params)
        self.model.fit(x.reshape(-1, n_features), y.reshape(-1).astype(np.float64))
        return self

    def decision_scores(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        n_shots, n_slots, n_features = x.shape
        return self.model.predict(x.reshape(-1, n_features)).reshape(n_shots, n_slots)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return _grouped_softmax(self.decision_scores(x))

    def importances(self, names: list[str]) -> dict[str, float]:
        """Gain-based importance by feature name, largest first."""
        if self.model is None:
            raise RuntimeError("model is not fitted")
        gains = self.model.booster_.feature_importance(importance_type="gain")
        total = gains.sum() or 1.0
        pairs = sorted(zip(names, gains / total, strict=True), key=lambda kv: -kv[1])
        return {name: float(value) for name, value in pairs}
