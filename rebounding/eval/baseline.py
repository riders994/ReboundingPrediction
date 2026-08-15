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

from rebounding.data.features import FEATURE_REGIMES, to_tensor
from rebounding.eval.metrics import Scores, evaluate
from rebounding.eval.split import Split, split_by_game
from rebounding.models.baselines import NearestPlayer, SlotPrior
from rebounding.models.conditional_logit import ConditionalLogit


def _distance_feature(regime: str) -> str:
    """The rim-distance column available in a regime, for the nearest-player rule."""
    return "pre_dist" if regime == "release" else "pos_dist"


def run(
    frame_path: str | Path = "data/frame.parquet",
    l2: float = 1.0,
    regimes: list[str] | None = None,
    with_forest: bool = False,
) -> tuple[pd.DataFrame, Split]:
    """Fit every model in every regime; return a tidy results frame and the split."""
    rows = pd.read_parquet(frame_path)
    split = split_by_game(rows)

    records = []
    for regime in regimes or list(FEATURE_REGIMES):
        names = FEATURE_REGIMES[regime]
        x_train, y_train, _ = to_tensor(split.train, names)
        x_val, y_val, _ = to_tensor(split.val, names)

        models = {
            "slot prior": SlotPrior(),
            f"nearest ({_distance_feature(regime)})": NearestPlayer(
                names, _distance_feature(regime)
            ),
            "conditional logit": ConditionalLogit(l2=l2),
        }
        if with_forest:
            from rebounding.models.forest import ForestRanker

            models["random forest"] = ForestRanker()

        for label, model in models.items():
            model.fit(x_train, y_train)
            scores: Scores = evaluate(model.predict_proba(x_val), y_val)
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

    logit = results[results["model"] == "conditional logit"].set_index("regime")["top1"]
    if {"release", "rim"} <= set(logit.index):
        gap = (logit["rim"] - logit["release"]) * 100
        lines.append(
            f"train/serve gap: rim {logit['rim']:.1%} - release {logit['release']:.1%} "
            f"= {gap:.1f} points. That is what a movement model would have to recover."
        )
    return "\n".join(lines)
