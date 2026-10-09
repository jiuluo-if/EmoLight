import numpy as np
import pytest

from emolight.emotion.predictor import ModelStatus
from emolight.emotion.sklearn_predictor import (
    SklearnEmotionPredictor,
    fit_calibrated_svm,
    save_model_artifact,
)
from emolight.events import Emotion
from emolight.features.prosody import (
    PROSODY_FEATURE_NAMES,
    PROSODY_SCHEMA_VERSION,
    ProsodyFeatures,
)


def synthetic_training_data():
    rng = np.random.default_rng(23)
    vectors, labels, groups = [], [], []
    emotions = [Emotion.NEUTRAL, Emotion.HAPPY, Emotion.ANGRY, Emotion.SAD]
    for speaker in range(12):
        for index, emotion in enumerate(emotions):
            vector = rng.normal(0.0, 0.08, size=32)
            vector[index] += 3.0
            vectors.append(vector)
            labels.append(emotion.value)
            groups.append(f"speaker-{speaker}")
    return np.asarray(vectors), np.asarray(labels), np.asarray(groups)


def example_features(values, sample_rate=16000, activity_rms_threshold=0.01):
    return ProsodyFeatures(
        values=values,
        feature_names=PROSODY_FEATURE_NAMES,
        schema_version=PROSODY_SCHEMA_VERSION,
        sample_rate=sample_rate,
        frame_ms=25.0,
        hop_ms=10.0,
        frame_length_samples=round(sample_rate * 0.025),
        hop_length_samples=round(sample_rate * 0.01),
        frame_count=100,
        voiced_fraction=0.8,
        activity_rms_threshold=activity_rms_threshold,
    )


def test_missing_or_corrupt_model_fails_closed(tmp_path):
    missing = SklearnEmotionPredictor.load(tmp_path / "missing.joblib")
    assert missing.predict(example_features(np.zeros(32))).model_status is ModelStatus.NOT_CONFIGURED

    corrupt_path = tmp_path / "corrupt.joblib"
    corrupt_path.write_bytes(b"not a joblib artifact")
    corrupt = SklearnEmotionPredictor.load(corrupt_path)
    assert corrupt.predict(example_features(np.zeros(32))).model_status is ModelStatus.INVALID


def test_calibrated_linear_svm_roundtrip_and_feature_contract(tmp_path):
    features, labels, groups = synthetic_training_data()
    artifact = fit_calibrated_svm(features, labels, groups, seed=9)
    model_path = tmp_path / "small-svm.joblib"
    size = save_model_artifact(model_path, artifact)
    predictor = SklearnEmotionPredictor.load(model_path)

    output = predictor.predict(example_features(features[0]))
    assert output.model_status is ModelStatus.READY
    assert output.emotion is Emotion.NEUTRAL
    assert set(output.scores) == set(Emotion)
    assert 0.0 <= output.confidence <= 1.0
    assert size == model_path.stat().st_size
    assert size < 100_000
    assert predictor.predict(example_features(features[0], sample_rate=8000)).model_status is ModelStatus.INVALID


def test_loader_rejects_schema_and_class_order_mismatch(tmp_path):
    joblib = pytest.importorskip("joblib")
    features, labels, groups = synthetic_training_data()
    artifact = fit_calibrated_svm(features, labels, groups, seed=3)

    schema_path = tmp_path / "schema.joblib"
    wrong_schema = dict(artifact)
    wrong_schema["metadata"] = dict(artifact["metadata"], feature_schema_version="prosody-v0")
    joblib.dump(wrong_schema, schema_path)
    assert SklearnEmotionPredictor.load(schema_path).predict(example_features(features[0])).model_status is ModelStatus.INVALID

    order_path = tmp_path / "classes.joblib"
    wrong_order = dict(artifact)
    wrong_order["metadata"] = dict(artifact["metadata"], classes=list(reversed(artifact["metadata"]["classes"])))
    joblib.dump(wrong_order, order_path)
    assert SklearnEmotionPredictor.load(order_path).predict(example_features(features[0])).model_status is ModelStatus.INVALID


def test_loader_rejects_frame_and_hop_mismatch(tmp_path):
    features, labels, groups = synthetic_training_data()
    artifact = fit_calibrated_svm(features, labels, groups, seed=11)
    path = tmp_path / "model.joblib"
    save_model_artifact(path, artifact)

    frame_mismatch = SklearnEmotionPredictor.load(path, expected_frame_ms=20.0)
    hop_mismatch = SklearnEmotionPredictor.load(path, expected_hop_ms=5.0)

    assert frame_mismatch.model_status is ModelStatus.INVALID
    assert hop_mismatch.model_status is ModelStatus.INVALID


def test_model_contract_includes_activity_threshold(tmp_path):
    features, labels, groups = synthetic_training_data()
    artifact = fit_calibrated_svm(
        features, labels, groups, seed=17, activity_rms_threshold=0.02,
    )
    path = tmp_path / "threshold.joblib"
    save_model_artifact(path, artifact)

    matching = SklearnEmotionPredictor.load(path, expected_activity_rms_threshold=0.02)
    mismatched = SklearnEmotionPredictor.load(path, expected_activity_rms_threshold=0.01)

    assert matching.model_status is ModelStatus.READY
    assert mismatched.model_status is ModelStatus.INVALID
    output = matching.predict(example_features(features[0], activity_rms_threshold=0.01))
    assert output.model_status is ModelStatus.INVALID


def test_calibration_cv_splits_never_share_speaker_groups():
    features, labels, groups = synthetic_training_data()
    artifact = fit_calibrated_svm(features, labels, groups, seed=5)

    for train_groups, calibration_groups in artifact["_calibration_group_audit"]:
        assert set(train_groups).isdisjoint(calibration_groups)
