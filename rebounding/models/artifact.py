"""The shipped artifact: the model, the fitted priors, and a record of both.

:mod:`rebounding.eval.baseline` fits every model in every regime and throws them all
away, which is what an experiment should do. This module builds the one model that
gets deployed and writes it to disk.

The artifact is a **bundle rather than a bare estimator**, because the served feature
set is not computable from the estimator alone. Seven of the twenty-seven columns in
:data:`~rebounding.data.features.SERVED_FEATURES` come out of
:class:`~rebounding.data.derived.ShotPriors`, whose damping constant and flight-time
regression are *fitted* on the games the model trained on. Pickling only the LightGBM
model would leave the serving side unable to build ``flight_hat`` at all, and nothing
would raise -- the app would quietly serve a model seven features short.

The weights are deliberately **not committed**: ``.gitignore`` still excludes
``*.pkl`` and the file reaches the web app by scp. That decision puts the burden here.
Nothing in git will record which model is on the box, so the bundle carries its own
provenance -- the commit it was built from, a hash of the corpus, the library versions
that wrote it, and the scores it earned -- and :meth:`ModelArtifact.describe` prints
it. "What is deployed?" is then answerable from the deployed file alone, which is the
question a scp workflow otherwise cannot answer.

**Serving hosts need ``rebounding`` importable.** The bundle references
:class:`~rebounding.models.boosted.BoostedSoftmax`, :class:`ShotPriors` and the
module-level ``grouped_softmax_objective`` by import path, so unpickling imports them.
The handoff brief already has the app importing this package's feature code rather
than retyping it, so this adds no dependency that was not already required.

Fitting the shipped model on ``train+val`` is the default. Hyperparameters were chosen
on the validation split, so once that choice is made, holding those 95 games out of
the final fit costs data for nothing. The test split stays untouched either way, and
the score in the metadata is measured on it -- so the number the app quotes belongs to
the weights the app is actually serving. ``--fit-on train`` reproduces the ladder in
the README instead.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rebounding.data.derived import ShotPriors
from rebounding.data.features import SERVED_FEATURES, to_tensor
from rebounding.eval.metrics import Scores, evaluate
from rebounding.eval.split import split_by_game
from rebounding.models.boosted import BoostedSoftmax

LOGGER = logging.getLogger(__name__)

# Bumped when the bundle's own layout changes in a way `load` cannot absorb.
ARTIFACT_VERSION = 1

# The name `webapp.py` loads. Kept so the scp lands on the path the app expects.
DEFAULT_OUTPUT = Path("FinalModel.pkl")

FIT_CHOICES = ("train", "train+val")

_TRACKED_PACKAGES = ("numpy", "pandas", "lightgbm", "scikit-learn", "joblib", "pyarrow")


def _git_provenance() -> dict[str, Any]:
    """The commit this was built from, and whether the tree was clean at the time.

    A dirty tree is recorded rather than refused: rebuilding weights mid-change is a
    normal thing to do. It matters only that the file says so, because a dirty build
    cannot be reproduced from the commit alone.
    """

    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                args, capture_output=True, text=True, timeout=10, cwd=Path(__file__).resolve().parent
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = run("git", "rev-parse", "HEAD")
    status = run("git", "status", "--porcelain")
    return {
        "commit": commit,
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        # None rather than False when git itself was unavailable, so "clean" and
        # "unknown" are not the same value.
        "dirty": None if status is None else bool(status),
    }


def _digest(path: Path) -> dict[str, Any]:
    """Size and SHA-256 of the corpus, so a deployed model names its own inputs."""
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return {"path": str(path), "sha256": sha.hexdigest(), "bytes": path.stat().st_size}


def _versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    found = {"python": sys.version.split()[0]}
    for name in _TRACKED_PACKAGES:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            continue
    return found


def _describe_rows(rows: pd.DataFrame) -> dict[str, int]:
    return {
        "games": int(rows["GameID"].nunique()),
        "shots": int(rows["ShotID"].nunique()),
        "rows": int(len(rows)),
    }


@dataclass
class ModelArtifact:
    """Everything the serving side needs, and everything ops needs to identify it."""

    model: BoostedSoftmax
    priors: ShotPriors
    features: list[str]
    regime: str
    metadata: dict[str, Any] = field(default_factory=dict)
    version: int = ARTIFACT_VERSION

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Per-player probabilities for ``(n_shots, 10, n_features)`` input.

        Already normalised within each shot by the grouped softmax. The old app
        divided per-row forest probabilities by their sum; doing that here would be a
        second normalisation of an already-normalised vector.
        """
        if x.shape[-1] != len(self.features):
            raise ValueError(
                f"expected {len(self.features)} features in {self.regime} order, got {x.shape[-1]}"
            )
        return self.model.predict_proba(x)

    def describe(self) -> str:
        meta = self.metadata
        git = meta.get("git", {})
        corpus = meta.get("corpus", {})
        dirty = {True: " (dirty)", False: "", None: " (unknown)"}[git.get("dirty")]

        lines = [
            f"artifact v{self.version}   regime {self.regime}   {len(self.features)} features",
            f"built    {meta.get('created_utc', '?')}",
            f"commit   {(git.get('commit') or '?')[:12]}{dirty} on {git.get('branch') or '?'}",
            f"corpus   {corpus.get('path', '?')}  sha256 {(corpus.get('sha256') or '?')[:12]}",
            f"fit on   {meta.get('fit_on', '?')}  "
            f"{meta.get('fit_rows', {}).get('games', '?')} games, "
            f"{meta.get('fit_rows', {}).get('shots', '?'):,} shots",
        ]
        for split_name, scores in (meta.get("scores") or {}).items():
            lines.append(
                f"{split_name:8} top1 {scores['top1']:6.1%}  top3 {scores['top3']:6.1%}  "
                f"mrr {scores['mrr']:.3f}  logloss {scores['log_loss']:.3f}  "
                f"n={scores['n_shots']:,}"
            )
        versions = meta.get("versions", {})
        lines.append("versions " + "  ".join(f"{k} {v}" for k, v in versions.items()))
        return "\n".join(lines)


def build(
    frame_path: str | Path = "data/frame.parquet",
    fit_on: str = "train+val",
    features: list[str] | None = None,
    regime: str = "served",
    model: BoostedSoftmax | None = None,
) -> ModelArtifact:
    """Fit the deployable model and wrap it with its priors and its provenance.

    ``ShotPriors`` is fitted on exactly the rows the model is fitted on, never on the
    test split -- otherwise the held-out games would contribute to their own features
    and the score in the metadata would be an overstatement of what the app can do.
    """
    if fit_on not in FIT_CHOICES:
        raise ValueError(f"fit_on must be one of {FIT_CHOICES}, got {fit_on!r}")

    frame_path = Path(frame_path)
    features = list(features or SERVED_FEATURES)
    split = split_by_game(pd.read_parquet(frame_path))

    fit_rows = pd.concat([split.train, split.val]) if fit_on == "train+val" else split.train
    priors = ShotPriors().fit(fit_rows)

    x_fit, y_fit, _ = to_tensor(priors.transform(fit_rows), features)
    model = (model or BoostedSoftmax()).fit(x_fit, y_fit)

    # Score on everything the model did not see. With fit_on="train" that includes
    # the validation split, which is what reproduces the README's ladder.
    held_out = {"test": split.test} if fit_on == "train+val" else {"val": split.val, "test": split.test}
    scores: dict[str, Scores] = {}
    for name, rows in held_out.items():
        x, y, _ = to_tensor(priors.transform(rows), features)
        scores[name] = evaluate(model.predict_proba(x), y)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": _git_provenance(),
        "corpus": _digest(frame_path),
        "fit_on": fit_on,
        "fit_rows": _describe_rows(fit_rows),
        "held_out_rows": {name: _describe_rows(rows) for name, rows in held_out.items()},
        "scores": {name: asdict(score) for name, score in scores.items()},
        "model": {"class": type(model).__name__, "params": _serialisable_params(model)},
        "versions": _versions(),
    }
    return ModelArtifact(
        model=model, priors=priors, features=features, regime=regime, metadata=metadata
    )


def _serialisable_params(model: BoostedSoftmax) -> dict[str, Any]:
    """Hyperparameters without the objective, which is a function and not data."""
    return {k: v for k, v in getattr(model, "params", {}).items() if not callable(v)}


def save(artifact: ModelArtifact, path: str | Path = DEFAULT_OUTPUT) -> Path:
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path, compress=3)
    return path


def load(path: str | Path = DEFAULT_OUTPUT) -> ModelArtifact:
    """Load a bundle, and say something useful when the environment cannot serve it."""
    import joblib

    path = Path(path)
    try:
        artifact = joblib.load(path)
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on the host
        raise ModuleNotFoundError(
            f"{path} references {exc.name!r}, which is not importable here. The bundle "
            "stores the model and priors by class, so the serving host needs the "
            "`rebounding` package installed (and lightgbm alongside it)."
        ) from exc

    if not isinstance(artifact, ModelArtifact):
        raise TypeError(f"{path} holds {type(artifact).__name__}, not a ModelArtifact")
    if artifact.version != ARTIFACT_VERSION:
        raise ValueError(
            f"{path} is artifact v{artifact.version}, this package writes v{ARTIFACT_VERSION}"
        )

    # The artifact's own list is authoritative -- it is the order the trees were fitted
    # in. A drifted package is the thing worth shouting about, because the failure is
    # silent: every feature still has a value, just the wrong one.
    if artifact.regime == "served" and artifact.features != SERVED_FEATURES:
        LOGGER.warning(
            "%s was fitted on a different feature list than this package's SERVED_FEATURES; "
            "serve artifact.features (%d columns), not the imported list (%d)",
            path, len(artifact.features), len(SERVED_FEATURES),
        )
    return artifact
