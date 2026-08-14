"""Grouped softmax over the ten players on the floor.

The question is "which of these ten gets the board", so the model scores each player
and normalises across the ten:

    P(player i) = exp(w . x_i) / sum_j exp(w . x_j)

This is a conditional logit. It is written out here rather than assembled from
``sklearn`` because sklearn's ``LogisticRegression`` has no grouped form: the
alternative is ten independent binary fits whose probabilities do not sum to one
over the group, which then need calibrating and class weighting before an argmax can
be taken. Those are exactly the hacks the original pipeline needed. Optimising the
grouped likelihood directly makes the training objective the same thing as the
evaluation metric, and the class imbalance disappears -- one positive in ten is the
structure of the problem, not a skew to be corrected.

Two consequences of the softmax being *within* a shot are worth stating:

* A global intercept cancels out, so there isn't one. Per-slot intercepts do not
  cancel, and ``fit_slot_bias`` controls them. With slot bias alone and no features,
  the model reduces to the empirical rate per slot -- which is the honest floor any
  feature set has to beat, and is available as :class:`SlotPrior`.
* Features constant across all ten players of a shot contribute nothing, for the
  same reason. Per-shot quantities like flight time only become usable if they are
  interacted with something that varies per player.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

N_PLAYERS = 10


def _log_softmax(scores: np.ndarray) -> np.ndarray:
    """Row-wise log softmax, shifted for numerical stability."""
    shifted = scores - scores.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


class ConditionalLogit:
    """Softmax over the ten players of a shot, fitted by L-BFGS on the exact NLL.

    ``x`` is ``(n_shots, 10, n_features)`` and ``y`` is ``(n_shots, 10)`` one-hot.
    Features are standardised on the training set; the same shift and scale are
    reused at predict time, so a split's statistics never leak into another's.
    """

    def __init__(self, l2: float = 1.0, fit_slot_bias: bool = True, max_iter: int = 500) -> None:
        self.l2 = l2
        self.fit_slot_bias = fit_slot_bias
        self.max_iter = max_iter
        self.weights: np.ndarray | None = None
        self.slot_bias: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def _standardise(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.scale_

    def _unpack(self, theta: np.ndarray, n_features: int) -> tuple[np.ndarray, np.ndarray]:
        weights = theta[:n_features]
        bias = theta[n_features:] if self.fit_slot_bias else np.zeros(N_PLAYERS)
        return weights, bias

    def fit(self, x: np.ndarray, y: np.ndarray) -> ConditionalLogit:
        n_shots, n_slots, n_features = x.shape
        if n_slots != N_PLAYERS:
            raise ValueError(f"expected {N_PLAYERS} players per shot, got {n_slots}")

        flat = x.reshape(-1, n_features)
        self.mean_ = flat.mean(axis=0)
        # Guard constant columns; a zero scale would produce NaNs rather than a
        # harmless all-zero feature.
        self.scale_ = np.where(flat.std(axis=0) > 1e-8, flat.std(axis=0), 1.0)
        xs = self._standardise(x)

        y = y.astype(np.float64)
        n_params = n_features + (N_PLAYERS if self.fit_slot_bias else 0)

        def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
            weights, bias = self._unpack(theta, n_features)
            scores = xs @ weights + bias  # (n_shots, 10)
            log_p = _log_softmax(scores)
            nll = -(y * log_p).sum() / n_shots

            residual = np.exp(log_p) - y  # (n_shots, 10)
            grad_w = np.einsum("sp,spf->f", residual, xs) / n_shots
            # L2 on the weights only. Penalising the slot bias would pull the model
            # away from the base rates, which are not noise here.
            grad_w += self.l2 * weights / n_shots
            nll += 0.5 * self.l2 * (weights @ weights) / n_shots

            if not self.fit_slot_bias:
                return nll, grad_w
            grad_b = residual.sum(axis=0) / n_shots
            return nll, np.concatenate([grad_w, grad_b])

        result = minimize(
            objective,
            np.zeros(n_params),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter},
        )
        self.weights, self.slot_bias = self._unpack(result.x, n_features)
        self.converged_ = bool(result.success)
        self.n_iter_ = int(result.nit)
        return self

    def decision_scores(self, x: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("model is not fitted")
        return self._standardise(x) @ self.weights + self.slot_bias

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.exp(_log_softmax(self.decision_scores(x)))

    def coefficients(self, names: list[str]) -> dict[str, float]:
        """Standardised coefficients by feature name, largest magnitude first."""
        if self.weights is None:
            raise RuntimeError("model is not fitted")
        pairs = sorted(zip(names, self.weights, strict=True), key=lambda kv: -abs(kv[1]))
        return {name: float(value) for name, value in pairs}
