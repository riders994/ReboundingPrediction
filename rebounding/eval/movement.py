"""Scoring the movement model on something other than squared error.

Squared error is the metric that produced the failure this model exists to fix, so it
cannot also be the metric that says whether the fix worked. A model that predicts the
conditional mean of a bimodal target wins on mean displacement error and loses on
everything a viewer can see. All four families below exist because one of them would
have caught the 2017 model and the obvious one would not.

**Reference error.** :func:`displacement_error` is kept, reported, and never optimised
against. Its job is to make the trade explicit: a calibrated model should be *worse*
here than a point model, and if it is not, the point model is underfitted.

**Sharpness against coverage.** ``min_ade`` asks whether the truth is anywhere in the
predicted set -- the standard minimum-over-K displacement error from the trajectory
literature -- and :func:`radial_calibration` asks whether the spread is honest. A model
can win the first by predicting everything and the second by predicting nothing; the
pair is only passable together.

**Scene plausibility.** Sampling ten players independently produces arrangements that
never occur: two bodies in the same square foot, a player covering forty feet in a
second and a half. These are measured against the same statistics computed on the real
rim-time frames, so the bar is "looks like the corpus", not a number somebody chose.

**Intention.** :func:`crash_rates` is the direct test of the multimodality claim.
Offence is close to a coin flip between crashing the glass and leaking out; a
conditional mean reproduces neither and lands between them. A model that recovers the
real rate is modelling the decision rather than averaging over it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rebounding.constants import HOOP

N_PLAYERS = 10

# Two players closer than this are effectively occupying one body. Chosen from the
# corpus rather than from anatomy: at rim time only 0.6% of real pairs are inside it.
CONTACT_FEET = 2.0

# Sustained speeds above this do not occur in the corpus, so a sample that needs one
# is a sample of something that cannot happen.
MAX_SPEED_FPS = 22.0


def _norm(delta: np.ndarray) -> np.ndarray:
    """Euclidean length along the last axis, which is always ``(x, y)`` here."""
    return np.hypot(delta[..., 0], delta[..., 1])


def displacement_error(predicted: np.ndarray, truth: np.ndarray) -> float:
    """Mean per-player L2 error in feet. Reference only -- do not tune on it."""
    return float(_norm(predicted - truth).mean())


def min_ade(samples: np.ndarray, truth: np.ndarray, scene: bool = False) -> float:
    """Best of ``n`` samples, in feet.

    ``samples`` is ``(n, n_shots, 10, 2)``. With ``scene=False`` each player is scored
    against his own best sample, which is the usual per-agent minADE and is generous:
    it lets a model assemble its answer from ten different futures. With ``scene=True``
    one sample has to be best for all ten at once, which is the number that matters for
    a model whose output is drawn as a single frame.
    """
    error = _norm(samples - truth[None])  # (n, shots, 10)
    return float(error.min(axis=0).mean() if not scene else error.mean(axis=2).min(axis=0).mean())


def radial_calibration(samples: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Is the predicted spread honest?

    For each player, rank the truth's distance from the sample centroid against the
    samples' own distances from it. Under a calibrated model that rank is uniform, so
    the 50% and 90% figures below should come out at 0.50 and 0.90. Under a confident
    model the truth is routinely further out than every sample and both collapse toward
    zero -- which is precisely the reading a point predictor gets, since all its
    samples sit on top of each other.

    ``spread`` is the mean sample-to-centroid distance, reported alongside because
    coverage is trivially perfect for a model that predicts an enormous cloud.
    """
    centroid = samples.mean(axis=0, keepdims=True)
    sample_radius = _norm(samples - centroid)  # (n, shots, 10)
    truth_radius = _norm(truth - centroid[0])  # (shots, 10)
    rank = (sample_radius > truth_radius[None]).mean(axis=0)
    return {
        "coverage_50": float((rank > 0.50).mean()),
        "coverage_90": float((rank > 0.10).mean()),
        "mean_rank": float(rank.mean()),
        "spread_ft": float(sample_radius.mean()),
    }


def _pairwise_min(positions: np.ndarray) -> np.ndarray:
    """Closest other player, per player. ``positions`` is ``(..., 10, 2)``."""
    delta = positions[..., :, None, :] - positions[..., None, :, :]
    separation = _norm(delta)
    eye = np.eye(N_PLAYERS, dtype=bool)
    return np.where(eye, np.inf, separation).min(axis=-1)


def plausibility(positions: np.ndarray, flight: np.ndarray, release: np.ndarray) -> dict[str, float]:
    """Could this arrangement of ten players actually happen?

    ``positions`` is ``(..., 10, 2)`` of absolute rim-time positions, ``release`` the
    matching release-time ones, and ``flight`` the per-shot flight time in seconds.
    Works on real frames and on sampled ones, which is the point: every number here is
    meant to be read next to the same number computed on the corpus.
    """
    closest = _pairwise_min(positions)
    travelled = _norm(positions - release)
    speed = travelled / np.maximum(flight, 1e-6)
    return {
        "min_separation_ft": float(closest.min(axis=-1).mean()),
        "contacts_per_scene": float((closest < CONTACT_FEET).sum(axis=-1).mean() / 2.0),
        "mean_speed_fps": float(speed.mean()),
        "impossible_speed_rate": float((speed > MAX_SPEED_FPS).mean()),
    }


def crash_rates(positions: np.ndarray, release: np.ndarray, is_offense: np.ndarray) -> dict[str, float]:
    """Fraction of players who end up closer to the rim than they started.

    The single most legible symptom of averaging a bimodal target. Real offence sits
    near a coin flip; a conditional-mean model pushes it hard to one side because the
    mean of "crash" and "leak" is a small drift in whichever direction is commoner.
    """
    def closing(xy):
        before = _norm(release - np.array(HOOP))
        after = _norm(xy - np.array(HOOP))
        return after < before

    closed = closing(positions)
    offense = is_offense.astype(bool)
    while offense.ndim < closed.ndim:
        offense = offense[None]
    offense = np.broadcast_to(offense, closed.shape)
    return {
        "crash_offense": float(closed[offense].mean()),
        "crash_defense": float(closed[~offense].mean()),
        "crash_all": float(closed.mean()),
    }


@dataclass(frozen=True)
class MovementScores:
    """Everything worth reporting about one movement model, on one split."""

    name: str
    n_shots: int
    error_ft: float
    min_ade_player: float
    min_ade_scene: float
    calibration: dict[str, float]
    plausibility: dict[str, float]
    crash: dict[str, float]

    def __str__(self) -> str:
        return (
            f"{self.name:<28} err {self.error_ft:5.2f}ft  "
            f"minADE p/s {self.min_ade_player:5.2f}/{self.min_ade_scene:5.2f}  "
            f"cov50 {self.calibration['coverage_50']:5.1%}  "
            f"cov90 {self.calibration['coverage_90']:5.1%}  "
            f"contacts {self.plausibility['contacts_per_scene']:4.2f}  "
            f"crash off/def {self.crash['crash_offense']:5.1%}/{self.crash['crash_defense']:5.1%}"
        )


def score(
    name: str,
    samples: np.ndarray,
    mean_prediction: np.ndarray,
    truth: np.ndarray,
    release: np.ndarray,
    flight: np.ndarray,
    is_offense: np.ndarray,
) -> MovementScores:
    """Assemble the full report. All displacements, all in feet.

    ``samples`` and ``mean_prediction`` are *displacements*; ``truth`` is the real
    displacement; ``release`` is absolute release-time position. Positions are
    reconstructed here rather than passed in so the two can never disagree.
    """
    return MovementScores(
        name=name,
        n_shots=len(truth),
        error_ft=displacement_error(mean_prediction, truth),
        min_ade_player=min_ade(samples, truth),
        min_ade_scene=min_ade(samples, truth, scene=True),
        calibration=radial_calibration(samples, truth),
        plausibility=plausibility(release[None] + samples, flight[None], release[None]),
        crash=crash_rates(release[None] + samples, release[None], is_offense),
    )
