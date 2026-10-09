import builtins
import json
from dataclasses import replace

import numpy as np

from emolight.audio.wav import AudioBuffer
from emolight.events import Emotion
from emolight.features.emotion_v1 import (
    EMOTION_FEATURE_NAMES,
    EMOTION_FEATURE_SCHEMA_VERSION,
    extract_emotion_features,
)
from emolight.emotion.predictor import ModelStatus
from emolight.emotion.linear import NumpyLinearEmotionPredictor


def example_artifact(*, threshold=0.65, calibration_status="CALIBRATED"):
    weights = np.zeros((2, len(EMOTION_FEATURE_NAMES)), dtype=float)
    weights[0, 0] = -0.02
    weights[1, 0] = 0.02
    return {
        "format_version": 1,
        "model_version": "emolight-linear-numpy-v1",
        "feature_profile": "full",
        "feature_schema_version": EMOTION_FEATURE_SCHEMA_VERSION,
        "feature_names": list(EMOTION_FEATURE_NAMES),
        "sample_rate": 16000,
        "frame_ms": 25.0,
        "hop_ms": 10.0,
        "activity_rms_threshold": 0.01,
        "quality_gates": {
            "min_activity_ratio": 0.2,
            "min_contiguous_active_frames": 3,
            "clipping_limit": 0.05,
            "minimum_snr_db": 5.0,
            "acceptable_snr_db": 10.0,
        },
        "classes": ["neutral", "happy"],
        "imputation_values": [200.0] + [0.0] * (len(EMOTION_FEATURE_NAMES) - 1),
        "standardization_mean": [0.0] * len(EMOTION_FEATURE_NAMES),
        "standardization_scale": [1.0] * len(EMOTION_FEATURE_NAMES),
        "weights": weights.tolist(),
        "intercepts": [3.0, -3.0],
        "calibration_slopes": [1.0, 1.0],
        "calibration_intercepts": [0.0, 0.0],
        "rejection_threshold": threshold,
        "calibration_status": calibration_status,
        "calibration_method": "one-vs-rest sigmoid on independent validation split",
        "training_library": {
            "scikit_learn_version": "test",
            "classifier": "LinearSVC",
            "C": 1.0,
            "dual": "auto",
            "random_state": 42,
            "standardization": "training split only",
            "imputation": "per-feature training median; fail if a column has no observations",
        },
        "training_manifest_sha256": "a" * 64,
        "label_mapping": {"happy": "happy", "neutral": "neutral"},
        "label_mapping_sha256": "b" * 64,
        "training": {
            "sample_count": 40,
            "class_counts": {"neutral": 20, "happy": 20},
        },
        "dataset_split": {
            "seed": 42,
            "method": "stratified group split",
            "record_counts": {"train": 40, "validation": 20, "test": 20},
            "subsets": {
                "train": {"group_count": 8},
                "validation": {"group_count": 4},
                "test": {"group_count": 4},
            },
        },
        "validation": {
            "sample_count": 20,
            "manifest_sha256": "a" * 64,
            "threshold_selected_on": "validation",
            "threshold_selection_method": "maximum F1 for correctness on validation",
            "class_counts": {"neutral": 10, "happy": 10},
            "brier_score": 0.12,
            "ece": 0.08,
        },
    }


def example_features():
    times = np.arange(16000) / 16000
    samples = (0.12 * np.sin(2.0 * np.pi * 220.0 * times)).astype(np.float32)
    return extract_emotion_features(AudioBuffer(samples, 16000))


def test_json_numpy_model_loads_and_predicts_with_validation_metadata(tmp_path):
    path = tmp_path / "emotion.json"
    artifact = example_artifact()
    path.write_text(json.dumps(artifact), encoding="utf-8")

    predictor = NumpyLinearEmotionPredictor.load(path, expected_sample_rate=16000)
    result = predictor.predict(example_features(), timestamp_ms=123)

    assert predictor.model_status is ModelStatus.READY
    assert result.emotion is Emotion.HAPPY
    assert result.calibration_status == "CALIBRATED"
    assert result.model_version == "emolight-linear-numpy-v1"
    assert result.timestamp_ms == 123
    assert result.scores[Emotion.HAPPY] > result.scores[Emotion.NEUTRAL]
    assert 0.0 <= result.confidence <= 1.0


def test_unconfigured_invalid_and_uncalibrated_artifacts_fail_closed(tmp_path):
    missing = NumpyLinearEmotionPredictor.load(tmp_path / "missing.json")
    corrupt_path = tmp_path / "corrupt.json"
    corrupt_path.write_text("{bad", encoding="utf-8")
    corrupt = NumpyLinearEmotionPredictor.load(corrupt_path)
    uncalibrated_path = tmp_path / "uncalibrated.json"
    uncalibrated_path.write_text(json.dumps(example_artifact(calibration_status="UNCALIBRATED")), encoding="utf-8")
    uncalibrated = NumpyLinearEmotionPredictor.load(uncalibrated_path)

    assert missing.predict(example_features()).model_status is ModelStatus.NOT_CONFIGURED
    assert corrupt.predict(example_features()).model_status is ModelStatus.INVALID
    assert uncalibrated.model_status is ModelStatus.UNCALIBRATED
    assert uncalibrated.predict(example_features()).emotion is None


def test_schema_sampling_and_threshold_mismatch_rejects_prediction(tmp_path):
    path = tmp_path / "emotion.json"
    path.write_text(json.dumps(example_artifact()), encoding="utf-8")

    rate_mismatch = NumpyLinearEmotionPredictor.load(path, expected_sample_rate=8000)
    valid = NumpyLinearEmotionPredictor.load(path)
    mismatched_features = replace(example_features(), activity_rms_threshold=0.02)

    assert rate_mismatch.model_status is ModelStatus.INVALID
    assert valid.predict(mismatched_features).model_status is ModelStatus.INVALID


def test_validation_selected_threshold_rejects_low_confidence_and_reports_coverage(tmp_path):
    path = tmp_path / "strict.json"
    path.write_text(json.dumps(example_artifact(threshold=0.99)), encoding="utf-8")
    predictor = NumpyLinearEmotionPredictor.load(path)

    result = predictor.predict(example_features())

    assert result.emotion is None
    assert result.rejection_status.value == "UNCERTAIN"
    assert result.model_version == "emolight-linear-numpy-v1"


def test_predictor_module_imports_without_sklearn(monkeypatch):
    import importlib
    import emolight.emotion.linear as linear_module

    real_import = builtins.__import__

    def forbid_sklearn(name, *args, **kwargs):
        if name.startswith("sklearn"):
            raise AssertionError("NumPy inference must not import scikit-learn")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_sklearn)
    importlib.reload(linear_module)


def test_model_missing_training_or_validation_provenance_is_invalid(tmp_path):
    artifact = example_artifact()
    del artifact["validation"]["threshold_selected_on"]
    path = tmp_path / "bad-provenance.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    predictor = NumpyLinearEmotionPredictor.load(path)

    assert predictor.model_status is ModelStatus.INVALID


def test_model_without_reproducible_training_parameters_is_invalid(tmp_path):
    artifact = example_artifact()
    del artifact["training_library"]
    path = tmp_path / "missing-training-metadata.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    assert NumpyLinearEmotionPredictor.load(path).model_status is ModelStatus.INVALID
