"""Random forest scored the same way as everything else.

This exists to answer one question: is the accuracy the conditional logit reaches a
limit of the *model* or of the *data*? A linear score over ten players cannot
express "this defender is boxed out by that one", and rebounding is full of such
interactions, so a tree ensemble is the obvious check before concluding anything
about the ceiling.

It also reproduces the shape of the 2017 model, which the README records as a random
forest reaching 86% top-1. That figure is not reproducible here and the arithmetic
suggests it was never top-1: a per-row binary classifier that predicts "not the
rebounder" for all ten players scores 90% row accuracy by construction, since nine
of ten rows are negative. 86% therefore sits *below* the trivial row-level baseline,
which is the signature of a per-row metric rather than a per-shot one. Fitting the
same model family and scoring it per shot is what makes the two comparable.

The forest is fitted per row and then ranked within a shot, which is exactly the
formulation :mod:`rebounding.models.conditional_logit` avoids -- the probabilities
are not normalised over the group. That is deliberate here: it is the comparison,
not the recommendation.
"""

from __future__ import annotations

import numpy as np

N_PLAYERS = 10


class ForestRanker:
    """Per-row binary random forest, ranked within each shot at prediction time."""

    def __init__(
        self,
        n_estimators: int = 300,
        min_samples_leaf: int = 20,
        max_depth: int | None = None,
        n_jobs: int = -1,
        random_state: int = 0,
    ) -> None:
        from sklearn.ensemble import RandomForestClassifier

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> ForestRanker:
        n_shots, n_slots, n_features = x.shape
        self.model.fit(x.reshape(-1, n_features), y.reshape(-1))
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        n_shots, n_slots, n_features = x.shape
        flat = self.model.predict_proba(x.reshape(-1, n_features))[:, 1]
        scores = flat.reshape(n_shots, n_slots)
        # Normalise within the shot so the numbers are comparable to the grouped
        # models. This is a renormalisation, not a calibration -- the forest was
        # never fitted against a per-shot likelihood.
        total = scores.sum(axis=1, keepdims=True)
        return np.divide(scores, total, out=np.full_like(scores, 1.0 / n_slots), where=total > 0)

    def row_accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        """Per-row binary accuracy: the metric the 2017 number was probably in.

        Reported so the comparison with the historical figure is explicit rather
        than implied. Predicting all-negative scores 0.9 here by construction.
        """
        n_shots, n_slots, n_features = x.shape
        predicted = self.model.predict(x.reshape(-1, n_features))
        return float((predicted == y.reshape(-1)).mean())
