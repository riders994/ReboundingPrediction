"""Tests for the deployable bundle.

The weights are gitignored and reach the web app by scp, so the bundle is the only
record of what is running. That shifts what is worth testing here. The model's
accuracy is `eval/baseline.py`'s problem; this file is about the two ways a bundle
goes wrong silently:

* it arrives on the serving host **missing something it needs** -- most obviously the
  fitted priors, without which seven of the twenty-seven served features cannot be
  computed and no exception says so;
* it arrives **unidentifiable**, so nobody can tell which commit or which corpus
  produced the file sitting on the box.

The provenance block is therefore asserted as a contract, not as decoration.
"""

import json

import numpy as np
import pandas as pd
import pytest

from rebounding.data.features import SERVED_FEATURES
from rebounding.models import artifact as artifact_module
from rebounding.models.artifact import ARTIFACT_VERSION, ModelArtifact, build, load, save
from rebounding.models.boosted import BoostedSoftmax

pytest.importorskip("lightgbm")
pytest.importorskip("joblib")

N_PLAYERS = 10
N_GAMES = 40
SHOTS_PER_GAME = 8


def _synthetic_frame(seed: int = 0) -> pd.DataFrame:
    """A frame with every column the served path touches, and a learnable label.

    The rebounder is the offensive player nearest the rim, so the fit has real signal
    to find and a degenerate model is distinguishable from a working one.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for game in range(N_GAMES):
        for shot in range(SHOTS_PER_GAME):
            distance = rng.uniform(1.0, 28.0, N_PLAYERS)
            is_offense = np.array([1.0] * 5 + [0.0] * 5)
            is_shooter = np.zeros(N_PLAYERS)
            is_shooter[rng.integers(0, 5)] = 1.0

            offense_best = int(np.argmin(np.where(is_offense == 1.0, distance, np.inf)))
            rebounder = np.zeros(N_PLAYERS, dtype=int)
            rebounder[offense_best] = 1

            angle = rng.uniform(-np.pi, np.pi, N_PLAYERS)
            rows.append(
                pd.DataFrame(
                    {
                        "GameID": f"00215{game:05d}",
                        "ShotID": f"{game:03d}-{shot:03d}",
                        "Slot": np.arange(N_PLAYERS),
                        "pre_x": rng.uniform(0, 47, N_PLAYERS),
                        "pre_y": rng.uniform(0, 50, N_PLAYERS),
                        "pre_dist": distance,
                        "pre_angle": angle,
                        "pre_vx": rng.normal(0, 3, N_PLAYERS),
                        "pre_vy": rng.normal(0, 3, N_PLAYERS),
                        "pre_cos_shooter": np.cos(angle),
                        "pre_box": rng.integers(0, 3, N_PLAYERS).astype(float),
                        "pos_x": rng.uniform(0, 47, N_PLAYERS),
                        "pos_y": rng.uniform(0, 50, N_PLAYERS),
                        "is_offense": is_offense,
                        "is_shooter": is_shooter,
                        "role": rng.integers(1, 6, N_PLAYERS).astype(float),
                        "FlightTime": 1.2 + 0.03 * distance[offense_best],
                        "IsTeamRebound": False,
                        "Rebounder": rebounder,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="module")
def frame_path(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("corpus") / "frame.parquet"
    _synthetic_frame().to_parquet(path)
    return str(path)


@pytest.fixture(scope="module")
def built(frame_path) -> ModelArtifact:
    return build(frame_path, model=BoostedSoftmax(n_estimators=25, learning_rate=0.2))


# --------------------------------------------------------------------------- #
# What the bundle has to contain
# --------------------------------------------------------------------------- #


def test_carries_the_fitted_priors(built):
    """Without these the serving side cannot build `flight_hat` at all."""
    assert built.priors.damping_ is not None
    assert built.priors.flight_coefficients_ is not None
    assert len(built.priors.flight_coefficients_) == 3


def test_carries_the_feature_list_in_fitted_order(built):
    assert built.features == SERVED_FEATURES
    assert len(built.features) == 27


def test_survives_a_save_load_round_trip(built, tmp_path):
    x = np.zeros((3, N_PLAYERS, len(built.features)), dtype=np.float32)
    path = save(built, tmp_path / "FinalModel.pkl")
    reloaded = load(path)

    assert reloaded.features == built.features
    assert reloaded.priors.damping_ == pytest.approx(built.priors.damping_)
    assert np.allclose(reloaded.predict_proba(x), built.predict_proba(x))


def test_probabilities_are_normalised_within_the_shot(built):
    """A grouped softmax needs no renormalisation; the old app divided by the sum."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=(6, N_PLAYERS, len(built.features))).astype(np.float32)
    probabilities = built.predict_proba(x)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_rejects_input_of_the_wrong_width(built):
    """Positional feature order was §3.6 of the handoff brief; this is the guard."""
    with pytest.raises(ValueError, match="expected 27 features"):
        built.predict_proba(np.zeros((2, N_PLAYERS, 9), dtype=np.float32))


# --------------------------------------------------------------------------- #
# Provenance, which is all the identification a gitignored file gets
# --------------------------------------------------------------------------- #


def test_records_the_commit_and_the_corpus(built):
    metadata = built.metadata
    assert metadata["corpus"]["sha256"]
    assert metadata["corpus"]["bytes"] > 0
    assert "commit" in metadata["git"]
    assert metadata["created_utc"].endswith("+00:00")
    assert metadata["versions"]["lightgbm"]


def test_metadata_is_plain_data(built):
    """It has to survive being read by something that is not this package."""
    json.dumps(built.metadata)  # raises if the LightGBM objective leaked in
    assert "objective" not in built.metadata["model"]["params"]


def test_describe_names_the_file(built):
    described = built.describe()
    assert "regime served" in described
    assert "27 features" in described
    assert "test" in described


# --------------------------------------------------------------------------- #
# What was fitted on what
# --------------------------------------------------------------------------- #


def test_shipping_fit_uses_train_and_val_but_never_test(built):
    assert built.metadata["fit_on"] == "train+val"
    assert built.metadata["fit_rows"]["games"] == 34  # 28 train + 6 val of 40
    assert set(built.metadata["held_out_rows"]) == {"test"}
    assert set(built.metadata["scores"]) == {"test"}


def test_train_only_fit_holds_out_val_as_well(frame_path):
    reproduction = build(
        frame_path, fit_on="train", model=BoostedSoftmax(n_estimators=25, learning_rate=0.2)
    )
    assert reproduction.metadata["fit_rows"]["games"] == 28
    assert set(reproduction.metadata["scores"]) == {"val", "test"}


def test_rejects_an_unknown_fit_split(frame_path):
    with pytest.raises(ValueError, match="fit_on must be one of"):
        build(frame_path, fit_on="everything")


def test_learns_the_planted_signal(built):
    """A bundle that saved a degenerate model would pass every test above."""
    assert built.metadata["scores"]["test"]["top1"] > 0.5


# --------------------------------------------------------------------------- #
# Loading something that is not what it claims
# --------------------------------------------------------------------------- #


def test_load_rejects_a_foreign_pickle(tmp_path):
    import joblib

    path = tmp_path / "not-a-bundle.pkl"
    joblib.dump({"model": "surprise"}, path)
    with pytest.raises(TypeError, match="not a ModelArtifact"):
        load(path)


def test_load_rejects_a_future_artifact_version(built, tmp_path):
    path = tmp_path / "from-the-future.pkl"
    save(built, path)

    import joblib

    stale = joblib.load(path)
    stale.version = ARTIFACT_VERSION + 1
    joblib.dump(stale, path)
    with pytest.raises(ValueError, match="artifact v"):
        load(path)


def test_load_warns_when_the_package_has_drifted(built, tmp_path, caplog):
    """Silent failure otherwise: every feature still has a value, just the wrong one."""
    path = tmp_path / "drifted.pkl"
    save(built, path)

    import joblib

    drifted = joblib.load(path)
    drifted.features = drifted.features[:-1]
    joblib.dump(drifted, path)

    with caplog.at_level("WARNING", logger=artifact_module.__name__):
        load(path)
    assert "SERVED_FEATURES" in caplog.text
