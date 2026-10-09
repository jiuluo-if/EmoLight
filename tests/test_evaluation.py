import numpy as np
import pytest
import csv

from emolight.training.evaluate import compute_classification_metrics, evaluate_manifest


def test_metrics_report_macro_f1_uar_precision_recall_calibration_and_confusion():
    labels = ("neutral", "happy", "neutral", "happy")
    classes = ("neutral", "happy")
    probabilities = np.asarray([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])

    result = compute_classification_metrics(labels, probabilities, classes)

    assert result["macro_f1"] == pytest.approx(1.0)
    assert result["uar"] == pytest.approx(1.0)
    assert result["per_class"]["neutral"]["precision"] == pytest.approx(1.0)
    assert result["per_class"]["happy"]["recall"] == pytest.approx(1.0)
    assert result["confusion_matrix"] == [[2, 0], [0, 2]]
    assert result["rejection_rate"] == 0.0
    assert 0.0 <= result["calibration"]["expected_calibration_error"] <= 1.0


def test_metrics_count_quality_and_low_confidence_rejections():
    labels = ("neutral", "happy")
    probabilities = np.asarray([[0.55, 0.45], [0.49, 0.51]])

    result = compute_classification_metrics(
        labels,
        probabilities,
        ("neutral", "happy"),
        eligible=(False, True),
        confidence_threshold=0.8,
    )

    assert result["rejection_rate"] == 1.0
    assert result["accepted_count"] == 0
    assert result["macro_f1"] is None


def test_metrics_report_test_labels_missing_from_the_trained_model():
    result = compute_classification_metrics(
        ("neutral", "sad"),
        np.asarray([[0.9, 0.1], [0.8, 0.2]]),
        ("neutral", "happy"),
    )

    assert result["unsupported_truth_count"] == 1
    assert result["confusion_matrix_labels"] == ["neutral", "happy", "sad"]
    assert result["per_class"]["sad"]["recall"] == 0.0
    assert result["uar"] == pytest.approx(0.5)
    assert result["calibration"]["sample_count"] == 1


def test_evaluator_withholds_metrics_without_real_data_attestation(tmp_path):
    wav_path = tmp_path / "local.wav"
    wav_path.touch()
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "emotion", "speaker_id", "recording_id"))
        writer.writeheader()
        writer.writerow({"path": wav_path.name, "emotion": "happy", "speaker_id": "s1", "recording_id": "r1"})

    result = evaluate_manifest(manifest, tmp_path / "missing-model.joblib")

    assert result["status"] == "NOT_EVALUATED_NO_LABELED_DATA"
    assert result["real_performance_evaluated"] is False
    assert result["identity_status"] == "NOT_EVALUATED"
    assert "macro_f1" not in result
