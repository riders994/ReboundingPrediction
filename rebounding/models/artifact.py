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

**Three models, three regimes.** This module builds any of them; which one it is
depends on ``regime``:

* the **source of truth** -- as feature-rich as the corpus allows, including rim-time
  positions and velocity. ``regime="all"``, or ``"rim"`` for the positional ceiling.
  It is a reference, not a product: :mod:`rebounding.serve` refuses to serve it,
  because none of those features exist when a user is placing dots.
* the **web app model** -- restricted to what the UI can supply. ``regime="served"``
  today, and static-only by necessity rather than by design: once a movement model
  exists, its predicted rim-time positions unlock the ``pos_*`` and ``move_*``
  families for this model too, and the regime grows to match.
* the **movement model** itself, built by :func:`build_movement` into a
  :class:`MovementArtifact`. It has a different shape -- positions to positions
  rather than players to a probability -- and a different consumer, since its output
  is both drawn in the UI and fed back in as the extra features of the regime above.

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
from rebounding.data.features import FEATURE_REGIMES, to_tensor
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
    movement: MovementArtifact | None = None,
) -> ModelArtifact:
    """Fit the deployable model and wrap it with its priors and its provenance.

    ``ShotPriors`` is fitted on exactly the rows the model is fitted on, never on the
    test split -- otherwise the held-out games would contribute to their own features
    and the score in the metadata would be an overstatement of what the app can do.

    The ``served+movement`` regime needs a fitted :class:`MovementArtifact` and builds
    its extra ten columns from that model's **predictions**, on the training rows as
    well as the held-out ones. Reading them off the frame instead would take rim-time
    truth into training and leave the app serving forecasts to a model that had only
    ever seen the real thing -- the exact skew that produced 2017's 86%.
    """
    if fit_on not in FIT_CHOICES:
        raise ValueError(f"fit_on must be one of {FIT_CHOICES}, got {fit_on!r}")
    if features is None and regime not in FEATURE_REGIMES:
        raise ValueError(f"regime must be one of {sorted(FEATURE_REGIMES)}, got {regime!r}")
    if regime == "served+movement" and movement is None:
        raise ValueError(
            "regime 'served+movement' needs a movement artifact to generate its "
            "rim-time columns; build one with `train-movement` and pass it in"
        )

    frame_path = Path(frame_path)
    features = list(features if features is not None else FEATURE_REGIMES[regime])
    split = split_by_game(pd.read_parquet(frame_path))

    fit_rows = pd.concat([split.train, split.val]) if fit_on == "train+val" else split.train
    priors = ShotPriors().fit(fit_rows)

    def tensor(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Features and labels, with the movement block appended when asked for.

        One ``ShotPriors`` serves both models. The movement artifact carries its own,
        fitted on its own games, but the app builds the served frame once and hands
        the same array to both -- so this does too, rather than measuring a pipeline
        nobody is going to run.
        """
        transformed = priors.transform(rows)
        if movement is None or regime != "served+movement":
            return to_tensor(transformed, features)[:2]
        served, labels, _ = to_tensor(transformed, movement.features)
        release, _, _ = to_tensor(transformed, ["pre_x", "pre_y"])
        shooter, _, _ = to_tensor(transformed, ["is_shooter"])
        block = movement.rim_block(served, release, shooter[..., 0])
        return np.concatenate([served, block], axis=-1), labels

    x_fit, y_fit = tensor(fit_rows)
    model = (model or BoostedSoftmax()).fit(x_fit, y_fit)

    # Score on everything the model did not see. With fit_on="train" that includes
    # the validation split, which is what reproduces the README's ladder.
    held_out = {"test": split.test} if fit_on == "train+val" else {"val": split.val, "test": split.test}
    scores: dict[str, Scores] = {}
    for name, rows in held_out.items():
        x, y = tensor(rows)
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
        "movement": None if movement is None else {
            "head": getattr(movement.state.get("config"), "head", "?"),
            "built": movement.metadata.get("created_utc"),
            "commit": (movement.metadata.get("git") or {}).get("commit"),
        },
        "versions": _versions(),
    }
    return ModelArtifact(
        model=model, priors=priors, features=features, regime=regime, metadata=metadata
    )


def _serialisable_params(model: BoostedSoftmax) -> dict[str, Any]:
    """Hyperparameters without the objective, which is a function and not data."""
    return {k: v for k, v in getattr(model, "params", {}).items() if not callable(v)}


def save(artifact: ModelArtifact | MovementArtifact, path: str | Path = DEFAULT_OUTPUT) -> Path:
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
            "`rebounding` package and its serving extra: "
            "pip install 'rebounding[serve]' "
            "--extra-index-url https://download.pytorch.org/whl/cpu"
        ) from exc

    if not isinstance(artifact, (ModelArtifact, MovementArtifact)):
        raise TypeError(f"{path} holds {type(artifact).__name__}, not an artifact")
    if artifact.version != ARTIFACT_VERSION:
        raise ValueError(
            f"{path} is artifact v{artifact.version}, this package writes v{ARTIFACT_VERSION}"
        )
    if isinstance(artifact, MovementArtifact):
        return artifact

    # The artifact's own list is authoritative -- it is the order the trees were fitted
    # in. A drifted package is the thing worth shouting about, because the failure is
    # silent: every feature still has a value, just the wrong one.
    expected = FEATURE_REGIMES.get(artifact.regime)
    if expected is not None and artifact.features != expected:
        LOGGER.warning(
            "%s was fitted on a different feature list than this package's %r regime; "
            "use artifact.features (%d columns), not the imported list (%d)",
            path, artifact.regime, len(artifact.features), len(expected),
        )
    return artifact


# -- the movement model ----------------------------------------------------------

# The name the web app loads for the movement model, replacing the 2017 `posnn.h5`.
DEFAULT_MOVEMENT_OUTPUT = Path("MovementModel.pkl")

# Fraction of the fitting games held back to stop on when there is no separate
# validation split left to use.
STOPPING_FRACTION = 0.1


@dataclass
class MovementArtifact:
    """The movement model, its input contract, and the priors that build that input.

    It carries **weights and a config, not a pickled network**. The modules in
    :mod:`rebounding.models.movement` are defined inside a function so that importing
    the package does not require torch, which makes them unpicklable -- but storing a
    state dict is also what pytorch recommends regardless, and it means this bundle
    loads without needing the class definitions to match byte for byte.

    The features are :data:`~rebounding.data.features.SERVED_FEATURES`: the movement
    model consumes exactly what the web app can supply, because its whole purpose is
    to run where rim-time positions do not exist yet.
    """

    state: dict[str, Any]
    priors: ShotPriors
    features: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    version: int = ARTIFACT_VERSION

    def __post_init__(self) -> None:
        self._model = None

    @property
    def model(self):
        """The reconstructed network, built once on first use."""
        from rebounding.models.movement import MovementModel

        if getattr(self, "_model", None) is None:
            self._model = MovementModel.from_state(self.state)
        return self._model

    def _check(self, x: np.ndarray) -> None:
        if x.shape[-1] != len(self.features):
            raise ValueError(
                f"expected {len(self.features)} served features, got {x.shape[-1]}"
            )

    def predict_positions(self, x: np.ndarray, release_xy: np.ndarray) -> np.ndarray:
        """Conditional-mean rim-time positions, ``(n_shots, 10, 2)``.

        For the rebounder, not for the animation. Drawing this is the 2017 failure:
        the mean of "crashes the glass" and "leaks out" is a player standing in
        neither place. Use :meth:`sample_positions` for anything a user watches.
        """
        self._check(x)
        return release_xy + self.model.predict(x)

    def sample_positions(
        self, x: np.ndarray, release_xy: np.ndarray, n: int = 20, seed: int | None = None
    ) -> np.ndarray:
        """``n`` whole-scene futures, ``(n, n_shots, 10, 2)``. This is what to draw."""
        self._check(x)
        return release_xy[None] + self.model.sample(x, n=n, seed=seed)

    def rim_block(
        self, x: np.ndarray, release_xy: np.ndarray, is_shooter: np.ndarray,
        positions: np.ndarray | None = None,
    ) -> np.ndarray:
        """The :data:`~rebounding.data.features.MOVEMENT_SUPPLIED` columns, in order.

        Concatenating this onto a ``served`` tensor produces the ``served+movement``
        regime. Pass ``positions`` to featurise a specific sampled scene rather than
        the conditional mean.
        """
        from rebounding.data.features import MOVEMENT_SUPPLIED, rim_features

        if positions is None:
            positions = self.predict_positions(x, release_xy)
        columns = rim_features(release_xy, positions, is_shooter)
        return np.stack([columns[name] for name in MOVEMENT_SUPPLIED], axis=-1).astype(np.float32)

    def describe(self) -> str:
        meta = self.metadata
        git = meta.get("git", {})
        dirty = {True: " (dirty)", False: "", None: " (unknown)"}[git.get("dirty")]
        config = self.state.get("config")
        lines = [
            f"movement artifact v{self.version}   head {getattr(config, 'head', '?')}   "
            f"{len(self.features)} input features",
            f"built    {meta.get('created_utc', '?')}",
            f"commit   {(git.get('commit') or '?')[:12]}{dirty} on {git.get('branch') or '?'}",
            f"corpus   {meta.get('corpus', {}).get('path', '?')}  "
            f"sha256 {(meta.get('corpus', {}).get('sha256') or '?')[:12]}",
            f"fit on   {meta.get('fit_on', '?')}  "
            f"{meta.get('fit_rows', {}).get('games', '?')} games, "
            f"{meta.get('fit_rows', {}).get('shots', '?'):,} shots  "
            f"(stopped at epoch {meta.get('best_epoch', '?')})",
            f"weights  {sum(v.size for v in self.state['weights'].values()):,} parameters",
        ]
        for split_name, scores in (meta.get("scores") or {}).items():
            lines.append(f"{split_name:8} " + "  ".join(f"{k} {v:.3f}" for k, v in scores.items()))
        return "\n".join(lines)


def build_movement(
    frame_path: str | Path = "data/frame.parquet",
    fit_on: str = "train",
    config=None,
) -> MovementArtifact:
    """Fit the movement model and wrap it with its priors and provenance.

    ``fit_on="train"`` is the default here, where the rebounder's is ``"train+val"``.
    A network needs a held-out set to stop on, and folding the validation games into
    the fit would leave nothing to stop against; with ``"train+val"`` the last tenth of
    the fitting games is carved off chronologically for that job instead, so the extra
    data comes at the cost of a noisier stopping signal rather than of a silent leak.
    """
    from rebounding.data.features import SERVED_FEATURES
    from rebounding.models.movement import MovementConfig, MovementModel

    if fit_on not in FIT_CHOICES:
        raise ValueError(f"fit_on must be one of {FIT_CHOICES}, got {fit_on!r}")

    frame_path = Path(frame_path)
    split = split_by_game(pd.read_parquet(frame_path))
    config = config or MovementConfig()

    if fit_on == "train":
        fit_rows, stop_rows = split.train, split.val
    else:
        games = sorted(pd.concat([split.train, split.val])["GameID"].unique())
        cut = int(len(games) * (1 - STOPPING_FRACTION))
        combined = pd.concat([split.train, split.val])
        fit_rows = combined[combined["GameID"].isin(set(games[:cut]))]
        stop_rows = combined[combined["GameID"].isin(set(games[cut:]))]

    priors = ShotPriors().fit(fit_rows)

    def tensors(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        transformed = priors.transform(rows)
        x, _, _ = to_tensor(transformed, SERVED_FEATURES)
        move, _, _ = to_tensor(transformed, ["move_dx", "move_dy"])
        return x, move

    x_fit, y_fit = tensors(fit_rows)
    x_stop, y_stop = tensors(stop_rows)
    model = MovementModel(config).fit(x_fit, y_fit, validation=(x_stop, y_stop))

    scores = {}
    for name, rows in (("stopping", stop_rows), ("test", split.test)):
        x, y = tensors(rows)
        scores[name] = {
            "error_ft": float(np.hypot(*(model.predict(x) - y).transpose(2, 0, 1)).mean()),
            "travelled_ft": float(np.hypot(y[..., 0], y[..., 1]).mean()),
        }

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": _git_provenance(),
        "corpus": _digest(frame_path),
        "fit_on": fit_on,
        "fit_rows": _describe_rows(fit_rows),
        "stopping_rows": _describe_rows(stop_rows),
        "best_epoch": model.best_epoch,
        "best_val": model.best_val,
        "scores": scores,
        "config": asdict(config),
        "versions": {**_versions(), "torch": _torch_version()},
    }
    return MovementArtifact(
        state=model.state(), priors=priors, features=list(SERVED_FEATURES), metadata=metadata
    )


def _torch_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("torch")
    except PackageNotFoundError:  # pragma: no cover - environment dependent
        return "?"


def load_movement(path: str | Path = DEFAULT_MOVEMENT_OUTPUT) -> MovementArtifact:
    """Load a movement bundle, refusing a rebounder bundle by the same name."""
    artifact = load(path)
    if not isinstance(artifact, MovementArtifact):
        raise TypeError(f"{path} holds a {type(artifact).__name__}, not a MovementArtifact")
    return artifact
