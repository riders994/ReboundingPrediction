"""The baseline experiment: what does position alone predict, and from when?

Runs every model against every feature regime and prints one table. The comparison
that matters is **release** against **rim**:

* ``rim`` trains on where players stand when the ball arrives. That is the regime
  the 2017 work reported 86% top-1 in, and it is a ceiling rather than a product --
  at serving time those positions have not happened yet.
* ``release`` trains on where players stand when the shot leaves the hand, which is
  all the web app can ever supply.

The distance between those two rows is the entire value of a movement model. If it
is small, the scene-level generative model in the rebuild plan is not worth
building; if it is large, that number is the budget it has to earn back. Either way
this is cheaper to measure than to assume, and the 2017 pipeline never measured it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rebounding.data.derived import ShotPriors
from rebounding.data.features import FEATURE_REGIMES, to_tensor
from rebounding.eval.metrics import Scores, evaluate
from rebounding.eval.split import Split, split_by_game
from rebounding.models.baselines import NearestPlayer, SlotPrior
from rebounding.models.boosted import BoostedSoftmax
from rebounding.models.conditional_logit import ConditionalLogit

# Regimes this module cannot fit, and why. `served+movement` is fittable only with a
# trained movement model in hand: its rim-time columns have to come from that model's
# *predictions*, and the frame's own `pos_*` are the truth. Reading them off the frame
# would produce a number that looks like the app's and is not -- the ladder would quote
# a servable regime scoring like the rim-time ceiling. Build it with
# `cli train --regime served+movement --movement MovementModel.pkl` instead.
UNFITTABLE_REGIMES = {
    "served+movement": (
        "needs a MovementArtifact to generate its rim-time columns; reading them off "
        "the frame would train on truth and report it as a servable score. Use "
        "`python -m rebounding.cli train --regime served+movement` instead"
    )
}


def _distance_feature(names: list[str]) -> str:
    """The rim-distance column available in a regime, for the nearest-player rule."""
    return "pos_dist" if "pos_dist" in names else "pre_dist"


def prepare(frame_path: str | Path = "data/frame.parquet") -> Split:
    """Split by game, then add the derived columns using training games only.

    :class:`rebounding.data.derived.ShotPriors` learns a damping constant and a
    flight-time regression, so it is fitted on the training split and applied to all
    three. Fitting it on the whole frame would let the validation and test games
    contribute to their own features.
    """
    split = split_by_game(pd.read_parquet(frame_path))
    priors = ShotPriors().fit(split.train)
    return Split(*(priors.transform(getattr(split, part)) for part in ("train", "val", "test")))


def run(
    frame_path: str | Path = "data/frame.parquet",
    l2: float = 1.0,
    regimes: list[str] | None = None,
    with_forest: bool = False,
    on: str = "val",
) -> tuple[pd.DataFrame, Split]:
    """Fit every model in every regime; return a tidy results frame and the split."""
    split = prepare(frame_path)

    records = []
    for regime in regimes or [r for r in FEATURE_REGIMES if r not in UNFITTABLE_REGIMES]:
        if regime in UNFITTABLE_REGIMES:
            raise ValueError(f"regime {regime!r} {UNFITTABLE_REGIMES[regime]}")
        names = FEATURE_REGIMES[regime]
        x_train, y_train, _ = to_tensor(split.train, names)
        x_eval, y_eval, _ = to_tensor(getattr(split, on), names)

        models = {
            "slot prior": SlotPrior(),
            f"nearest ({_distance_feature(names)})": NearestPlayer(
                names, _distance_feature(names)
            ),
            "conditional logit": ConditionalLogit(l2=l2),
            "boosted softmax": BoostedSoftmax(),
        }
        if with_forest:
            from rebounding.models.forest import ForestRanker

            models["random forest"] = ForestRanker()

        for label, model in models.items():
            model.fit(x_train, y_train)
            scores: Scores = evaluate(model.predict_proba(x_eval), y_eval)
            records.append(
                {
                    "regime": regime,
                    "model": label,
                    "top1": scores.top1,
                    "top3": scores.top3,
                    "mrr": scores.mrr,
                    "log_loss": scores.log_loss,
                    "n_features": len(names),
                }
            )

    return pd.DataFrame.from_records(records), split


def format_results(results: pd.DataFrame) -> str:
    lines = [
        f"{'regime':9} {'model':26} {'top1':>7} {'top3':>7} {'mrr':>7} {'logloss':>8}",
        "-" * 68,
    ]
    for regime, group in results.groupby("regime", sort=False):
        for _, row in group.iterrows():
            lines.append(
                f"{regime:9} {row['model']:26} {row['top1']:6.1%} {row['top3']:6.1%} "
                f"{row['mrr']:7.3f} {row['log_loss']:8.3f}"
            )
        lines.append("")

    best = results.loc[results.groupby("regime", sort=False)["top1"].idxmax()].set_index("regime")
    if {"release+derived", "rim"} <= set(best.index):
        served = best.loc["served"] if "served" in best.index else best.loc["release+derived"]
        rim, release = best.loc["rim"], best.loc["release+derived"]
        lines.append(
            f"best per regime: rim {rim['top1']:.1%} ({rim['model']}), "
            f"release {release['top1']:.1%} ({release['model']}), "
            f"served {served['top1']:.1%} ({served['model']})"
        )
        lines.append(
            f"train/serve gap: {(rim['top1'] - release['top1']) * 100:.1f} points. "
            "That is what a movement model would have to recover."
        )
    return "\n".join(lines)
