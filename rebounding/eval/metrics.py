"""Evaluation metrics for the grouped, one-of-ten prediction.

Top-1 accuracy is the headline because it is the question the web app asks: given
where everyone is, who gets the board. Top-3 is reported alongside it because a
coaching tool that narrows ten players to three is useful even when it is not exact,
and because a large top-1/top-3 gap says the model is ranking sensibly while
struggling to separate the leaders.

Mean reciprocal rank is included as the ranking-quality summary that does not depend
on where an arbitrary cutoff falls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Scores:
    n_shots: int
    top1: float
    top3: float
    mrr: float
    log_loss: float

    def __str__(self) -> str:
        return (
            f"top1 {self.top1:6.1%}   top3 {self.top3:6.1%}   "
            f"mrr {self.mrr:.3f}   logloss {self.log_loss:.3f}   n={self.n_shots:,}"
        )


def _ranks_of_truth(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """1-based rank of the true player within each shot, ties broken pessimistically.

    Pessimistic ties matter for the distance baselines, where exact ties are
    possible: counting a tie as a win would report accuracy the model cannot deliver
    when it has to commit to one player.
    """
    truth = labels.argmax(axis=1)
    true_prob = probabilities[np.arange(len(labels)), truth]
    # Everything strictly ahead, plus every tie including the true player itself.
    # Ten identical probabilities therefore rank 10, not 1.
    strictly_ahead = (probabilities > true_prob[:, None]).sum(axis=1)
    tied = (probabilities == true_prob[:, None]).sum(axis=1)
    return strictly_ahead + tied


def evaluate(probabilities: np.ndarray, labels: np.ndarray) -> Scores:
    """Score predicted per-player probabilities against one-hot labels."""
    if probabilities.shape != labels.shape:
        raise ValueError(f"shape mismatch: {probabilities.shape} vs {labels.shape}")
    if not (labels.sum(axis=1) == 1).all():
        raise ValueError("every shot must have exactly one labelled rebounder")

    ranks = _ranks_of_truth(probabilities, labels)
    truth = labels.argmax(axis=1)
    true_prob = probabilities[np.arange(len(labels)), truth]

    return Scores(
        n_shots=len(labels),
        top1=float((ranks == 1).mean()),
        top3=float((ranks <= 3).mean()),
        mrr=float((1.0 / ranks).mean()),
        log_loss=float(-np.log(np.clip(true_prob, 1e-12, None)).mean()),
    )
