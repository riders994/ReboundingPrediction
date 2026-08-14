"""Floors that any real model has to clear.

These exist so that an accuracy number means something. The canonical slot ordering
in :mod:`rebounding.data.features` is itself informative -- slots are sorted by
distance to the rim at release, offense first -- so a model fed those features starts
with a large head start that has nothing to do with having learned anything about
rebounding. :class:`SlotPrior` measures exactly that head start, and
:class:`NearestPlayer` measures what a single distance column buys on top of it.

A conditional logit that does not clearly beat both is not learning position; it is
re-deriving the sort order it was handed.
"""

from __future__ import annotations

import numpy as np

N_PLAYERS = 10


class SlotPrior:
    """Predicts the slot that most often gets the rebound, ignoring all features.

    The empirical rate per slot, learned from the training split. Because the slots
    are distance-ordered this is not a trivial baseline: it already encodes "the
    nearest defender usually gets it".
    """

    def __init__(self) -> None:
        self.rates: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> SlotPrior:
        self.rates = y.mean(axis=0)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.rates is None:
            raise RuntimeError("model is not fitted")
        rates = self.rates / self.rates.sum()
        return np.tile(rates, (len(x), 1))


class NearestPlayer:
    """Predicts whoever is closest to the rim, by one named distance column.

    Scores are the negated distance, so the argmax is the nearest player. Ranking is
    by distance alone; ``predict_proba`` returns a softmax of the negated distance
    purely so it shares an interface with the other models, and its calibration is
    not meaningful.
    """

    def __init__(self, feature_names: list[str], distance_feature: str = "pos_dist") -> None:
        if distance_feature not in feature_names:
            raise ValueError(f"{distance_feature!r} not among the supplied features")
        self.index = feature_names.index(distance_feature)
        self.distance_feature = distance_feature

    def fit(self, x: np.ndarray, y: np.ndarray) -> NearestPlayer:
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        scores = -x[:, :, self.index]
        shifted = scores - scores.max(axis=1, keepdims=True)
        exponentiated = np.exp(shifted)
        return exponentiated / exponentiated.sum(axis=1, keepdims=True)
