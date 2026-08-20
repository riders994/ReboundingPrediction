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
            # Rim-time columns, so every regime -- not just the web app's -- is
            # buildable from this fixture.
            pos_distance = np.maximum(distance + rng.normal(0, 3, N_PLAYERS), 0.5)
            move = rng.normal(0, 4, (N_PLAYERS, 2))
            velocity = rng.normal(0, 3, (N_PLAYERS, 2))
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
                        "pre_vx": velocity[:, 0],
                        "pre_vy": velocity[:, 1],
                        "pre_speed": np.hypot(velocity[:, 0], velocity[:, 1]),
                        "pre_cos_shooter": np.cos(angle),
                        "pre_box": rng.integers(0, 3, N_PLAYERS).astype(float),
                        "pos_x": rng.uniform(0, 47, N_PLAYERS),
                        "pos_y": rng.uniform(0, 50, N_PLAYERS),
                        "pos_dist": pos_distance,
                        "pos_angle": rng.uniform(-np.pi, np.pi, N_PLAYERS),
                        "pos_cos_shooter": np.cos(angle),
                        "pos_box": rng.integers(0, 3, N_PLAYERS).astype(float),
                        "move_dx": move[:, 0],
                        "move_dy": move[:, 1],
                        "move_dist": np.linalg.norm(move, axis=1),
                        "closed_on_rim": np.where(pos_distance < distance, 1.0, -1.0),
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


def test_builds_the_source_of_truth_regime_too(frame_path):
    """Three models, and this module has to be able to build more than the web app's."""
    rich = build(
        frame_path, regime="rim", model=BoostedSoftmax(n_estimators=15, learning_rate=0.2)
    )
    assert rich.regime == "rim"
    assert "pos_x" in rich.features
    assert len(rich.features) != 27


def test_rejects_an_unknown_regime(frame_path):
    with pytest.raises(ValueError, match="regime must be one of"):
        build(frame_path, regime="wishful")


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
    # `load` now accepts either bundle -- the rebounder's or the movement model's --
    # so it names the family rather than one class.
    with pytest.raises(TypeError, match="not an artifact"):
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
    assert "artifact.features" in caplog.text
    assert "served" in caplog.text


# --------------------------------------------------------------------------- #
# The movement bundle, and the regime that depends on it
# --------------------------------------------------------------------------- #

torch = pytest.importorskip("torch", reason="the movement artifact needs torch")


@pytest.fixture(scope="module")
def movement_built(frame_path):
    from rebounding.models.artifact import build_movement
    from rebounding.models.movement import MovementConfig

    return build_movement(
        frame_path,
        config=MovementConfig(
            head="cvae", d_model=16, n_layers=1, n_heads=2, d_ff=32, d_latent=4,
            epochs=2, batch_size=64, patience=99,
        ),
    )


def test_the_movement_bundle_carries_weights_not_a_network(movement_built):
    """The modules live inside a closure and cannot be pickled; the weights can.

    This is also what pytorch recommends, so the constraint and the good practice
    happen to agree -- but the bundle would be silently unloadable if it drifted back
    to storing the object.
    """
    assert set(movement_built.state) == {"config", "n_features", "weights"}
    assert all(isinstance(w, np.ndarray) for w in movement_built.state["weights"].values())


def test_the_movement_bundle_round_trips(movement_built, tmp_path):
    from rebounding.models.artifact import load_movement

    path = save(movement_built, tmp_path / "MovementModel.pkl")
    reloaded = load_movement(path)

    x = np.zeros((4, N_PLAYERS, len(movement_built.features)), dtype=np.float32)
    release = np.tile(np.array([30.0, 25.0]), (4, N_PLAYERS, 1))
    np.testing.assert_allclose(
        reloaded.predict_positions(x, release), movement_built.predict_positions(x, release),
        atol=1e-5,
    )


def test_the_movement_bundle_consumes_the_served_features(movement_built):
    """It has to run where rim-time positions do not exist, so it eats what the app has."""
    assert movement_built.features == SERVED_FEATURES


def test_the_movement_bundle_refuses_the_wrong_width(movement_built):
    with pytest.raises(ValueError, match="27 served features"):
        movement_built.predict_positions(
            np.zeros((2, N_PLAYERS, 5), dtype=np.float32), np.zeros((2, N_PLAYERS, 2))
        )


def test_load_movement_refuses_a_rebounder_bundle(built, tmp_path):
    from rebounding.models.artifact import load_movement

    path = save(built, tmp_path / "FinalModel.pkl")
    with pytest.raises(TypeError, match="not a MovementArtifact"):
        load_movement(path)


def test_the_movement_regime_needs_a_movement_model(frame_path):
    with pytest.raises(ValueError, match="needs a movement artifact"):
        build(frame_path, regime="served+movement")


def test_the_movement_regime_trains_on_predictions_not_on_truth(frame_path, movement_built):
    """The rim-time columns must come from the model, never off the frame.

    Training on the corpus's real ``pos_*`` and serving forecasts is the skew that
    produced 2017's 86% top-1, so this asserts the arithmetic rather than trusting the
    comment above it: the fitted block has to match what the movement model predicts
    and *not* match the truth sitting in the same frame.
    """
    from rebounding.data.features import MOVEMENT_SUPPLIED, to_tensor
    from rebounding.eval.split import split_by_game

    artifact_out = build(
        frame_path, regime="served+movement", movement=movement_built,
        model=BoostedSoftmax(n_estimators=10, learning_rate=0.2),
    )
    assert artifact_out.features == [*SERVED_FEATURES, *MOVEMENT_SUPPLIED]
    assert artifact_out.metadata["movement"]["head"] == "cvae"

    rows = split_by_game(pd.read_parquet(frame_path)).test
    transformed = artifact_out.priors.transform(rows)
    served, _, _ = to_tensor(transformed, SERVED_FEATURES)
    release, _, _ = to_tensor(transformed, ["pre_x", "pre_y"])
    shooter, _, _ = to_tensor(transformed, ["is_shooter"])
    truth, _, _ = to_tensor(transformed, ["pos_x", "pos_y"])

    block = movement_built.rim_block(served, release, shooter[..., 0])
    predicted_positions = movement_built.predict_positions(served, release)
    np.testing.assert_allclose(block[..., :2], predicted_positions, rtol=1e-5)
    # The fixture's rim positions are random, so a model that had copied them would
    # have to be within tracking noise of them. It is nowhere near.
    assert np.abs(block[..., :2] - truth).mean() > 1.0
