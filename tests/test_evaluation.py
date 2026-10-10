import numpy as np
import pytest
import csv

from emolight.training.evaluate import _aggregate_recordings, _recording_bootstrap_intervals, _summarize_per_speaker, _without_public_speaker_identifiers, compute_classification_metrics, evaluate_manifest, evaluate_model_pair


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


def test_metrics_report_coverage_and_error_rate_after_rejections():
    result = compute_classification_metrics(
        ("neutral", "happy", "neutral", "sad"),
        np.asarray([
            [0.9, 0.1],
            [0.1, 0.9],
            [0.2, 0.8],
            [0.55, 0.45],
        ]),
        ("neutral", "happy"),
        confidence_threshold=0.6,
    )

    assert result["coverage"] == pytest.approx(0.75)
    assert result["rejection_rate"] == pytest.approx(0.25)
    assert result["accepted_error_count"] == 1
    assert result["accepted_error_rate"] == pytest.approx(1.0 / 3.0)


def test_recording_aggregation_averages_only_quality_eligible_windows():
    result = _aggregate_recordings(
        labels=("happy", "happy", "sad"),
        probabilities=np.asarray([[0.8, 0.2], [0.1, 0.9], [0.1, 0.9]]),
        eligible=(True, False, False),
        speaker_ids=("s1", "s1", "s2"),
        recording_keys=("r1", "r1", "r2"),
    )

    assert result["labels"] == ("happy", "sad")
    assert result["probabilities"].tolist() == [[0.8, 0.2], [0.5, 0.5]]
    assert result["eligible"] == (True, False)
    assert result["speaker_ids"] == ("s1", "s2")
    intervals = _recording_bootstrap_intervals(
        result, ("happy", "sad"), 0.5, seed=42, resamples=100
    )
    assert intervals["intervals"]["recall_happy"] is not None


def test_public_speaker_summary_reports_ranges_without_speaker_keys():
    summary = _summarize_per_speaker({
        "private-speaker-a": {"sample_count": 3, "macro_f1": 0.4, "uar": 0.5, "coverage": 1.0, "accepted_error_rate": 0.2, "per_class": {"happy": {"recall": 0.0}}},
        "private-speaker-b": {"sample_count": 5, "macro_f1": 0.8, "uar": 0.9, "coverage": 0.8, "accepted_error_rate": 0.1, "per_class": {"happy": {"recall": 0.5}}},
    })

    assert summary["speaker_count"] == 2
    assert summary["metric_ranges"]["happy_recall"] == [0.0, 0.5]
    assert "private-speaker-a" not in repr(summary)
    assert "private-speaker-b" not in repr(summary)


def test_public_provenance_removes_nested_speaker_identifier_fields():
    result = _without_public_speaker_identifiers({
        "dataset_id": "example",
        "split": {"split_speaker_ids": {"train": ["private-id"]}},
    })

    assert result == {"dataset_id": "example", "split": {}}


def test_evaluator_withholds_metrics_without_real_data_attestation(tmp_path):
    wav_path = tmp_path / "local.wav"
    wav_path.touch()
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "emotion", "speaker_id", "recording_id"))
        writer.writeheader()
        writer.writerow({"path": wav_path.name, "emotion": "happy", "speaker_id": "s1", "recording_id": "r1"})

    result = evaluate_manifest(manifest, tmp_path / "missing-model.json")

    assert result["status"] == "NOT_EVALUATED_NO_LABELED_DATA"
    assert result["real_performance_evaluated"] is False
    assert result["identity_status"] == "NOT_EVALUATED"
    assert "macro_f1" not in result


def test_model_pair_uses_identical_evaluation_inputs_and_reports_deltas(monkeypatch):
    import emolight.training.evaluate as evaluation

    calls = []

    def fake_evaluate(manifest_path, model_path, **kwargs):
        calls.append((manifest_path, model_path, kwargs))
        f1 = 0.6 if model_path == "simple.json" else 0.8
        return {
            "status": "EVALUATED_USER_ATTESTED_LABELED_DATA",
            "real_performance_evaluated": True,
            "conditions": {
                "clean": {"metrics": {"macro_f1": f1, "coverage": 0.75, "accepted_error_rate": 0.2}}
            },
        }

    monkeypatch.setattr(evaluation, "evaluate_manifest", fake_evaluate)
    result = evaluate_model_pair(
        "data.csv", "simple.json", "full.json", confirm_real_labeled_data=True, seed=12,
        noise_manifest_path="noise.csv",
    )

    assert [call[1] for call in calls] == ["simple.json", "full.json"]
    assert calls[0][0] == calls[1][0] == "data.csv"
    assert calls[0][2] == calls[1][2] == {
        "confirm_real_labeled_data": True, "seed": 12, "noise_manifest_path": "noise.csv"
    }
    assert result["comparison_protocol"]["same_manifest"] is True
    assert result["conditions"]["clean"]["delta_full_minus_simple"] == {
        "macro_f1": pytest.approx(0.2),
        "coverage": 0.0,
        "accepted_error_rate": 0.0,
    }
