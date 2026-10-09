import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Mapping, Sequence

import numpy as np

from emolight.audio.wav import load_wav, AudioBuffer
from emolight.data.manifest import load_manifest
from emolight.data.split import split_by_speaker_and_recording
from emolight.emotion.predictor import ModelStatus
from emolight.emotion.sklearn_predictor import SklearnEmotionPredictor
from emolight.events import Emotion
from emolight.features.acoustic import AudioQualityStatus, extract_features
from emolight.features.prosody import extract_prosody
from emolight.training.noise import add_interferer_sir, add_white_noise_snr


def compute_classification_metrics(
    true_labels: Sequence[str],
    probabilities: np.ndarray,
    classes: Sequence[str],
    *,
    eligible: Sequence[bool] | None = None,
    confidence_threshold: float = 0.5,
) -> dict:
    """Compute held-out metrics; callers must separately establish data provenance."""
    y_true = np.asarray(true_labels, dtype=str)
    probs = np.asarray(probabilities, dtype=np.float64)
    class_order = tuple(str(label) for label in classes)
    if probs.ndim != 2 or probs.shape != (y_true.size, len(class_order)):
        raise ValueError("probabilities must have shape (rows, classes)")
    if not y_true.size or len(set(class_order)) != len(class_order) or not class_order:
        raise ValueError("labels and unique classes must be non-empty")
    if not np.isfinite(probs).all() or np.any(probs < 0) or np.any(probs > 1):
        raise ValueError("probabilities must be finite values in [0, 1]")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("each probability row must sum to 1")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")
    valid = np.ones(y_true.size, dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
    if valid.shape != y_true.shape:
        raise ValueError("eligible mask must have one value per row")

    max_probability = np.max(probs, axis=1)
    predictions = np.asarray(class_order)[np.argmax(probs, axis=1)]
    accepted = valid & (max_probability >= confidence_threshold)
    accepted_labels = y_true[accepted]
    accepted_predictions = predictions[accepted]
    observed = tuple(sorted(set(y_true.tolist())))
    report_classes = tuple(class_order) + tuple(sorted(set(observed) - set(class_order)))
    confusion = np.zeros((len(report_classes), len(report_classes)), dtype=np.int64)
    per_class: dict[str, dict[str, float | int]] = {}
    if accepted.any():
        from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support, recall_score

        confusion = confusion_matrix(accepted_labels, accepted_predictions, labels=report_classes)
        precision, recall, _, support = precision_recall_fscore_support(
            accepted_labels, accepted_predictions, labels=report_classes, zero_division=0
        )
        for index, label in enumerate(report_classes):
            per_class[label] = {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "support": int(np.sum(accepted_labels == label)),
            }
        macro_f1 = float(f1_score(accepted_labels, accepted_predictions, labels=observed, average="macro", zero_division=0))
        uar = float(recall_score(accepted_labels, accepted_predictions, labels=observed, average="macro", zero_division=0))
    else:
        for label in report_classes:
            per_class[label] = {"precision": 0.0, "recall": 0.0, "support": 0}
        macro_f1 = None
        uar = None

    calibration_mask = valid & np.isin(y_true, class_order)
    calibration = _calibration_metrics(y_true[calibration_mask], probs[calibration_mask], class_order)
    return {
        "sample_count": int(y_true.size),
        "accepted_count": int(accepted.sum()),
        "rejection_rate": float(1.0 - accepted.mean()),
        "confidence_threshold": confidence_threshold,
        "macro_f1": macro_f1,
        "uar": uar,
        "per_class": per_class,
        "confusion_matrix_labels": list(report_classes),
        "confusion_matrix": confusion.astype(int).tolist(),
        "calibration": calibration,
        "unsupported_truth_count": int(np.sum(~np.isin(y_true, class_order))),
    }


def _calibration_metrics(labels: np.ndarray, probabilities: np.ndarray, classes: tuple[str, ...]) -> dict:
    if labels.size == 0:
        return {"sample_count": 0, "multiclass_brier_score": None, "expected_calibration_error": None}
    class_indices = {label: index for index, label in enumerate(classes)}
    one_hot = np.zeros_like(probabilities)
    for row, label in enumerate(labels):
        one_hot[row, class_indices[label]] = 1.0
    brier = float(np.mean(np.sum(np.square(probabilities - one_hot), axis=1)))
    confidence = np.max(probabilities, axis=1)
    predicted = np.asarray(classes)[np.argmax(probabilities, axis=1)]
    correct = (predicted == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for index in range(len(bins) - 1):
        in_bin = (confidence >= bins[index]) & (confidence < bins[index + 1] if index < len(bins) - 2 else confidence <= bins[index + 1])
        if in_bin.any():
            ece += float(np.mean(in_bin)) * abs(float(np.mean(correct[in_bin])) - float(np.mean(confidence[in_bin])))
    return {
        "sample_count": int(labels.size),
        "multiclass_brier_score": brier,
        "expected_calibration_error": float(ece),
    }


class _PeakRssSampler:
    def __init__(self, interval_s: float = 0.02) -> None:
        import psutil

        self._process = psutil.Process()
        self._interval_s = interval_s
        self._stop = threading.Event()
        self.peak_bytes = self._process.memory_info().rss
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()
        self.peak_bytes = max(self.peak_bytes, self._process.memory_info().rss)

    def _sample(self) -> None:
        while not self._stop.wait(self._interval_s):
            self.peak_bytes = max(self.peak_bytes, self._process.memory_info().rss)


def evaluate_manifest(
    manifest_path: str | Path,
    model_path: str | Path,
    *,
    confirm_real_labeled_data: bool = False,
    seed: int | None = None,
    confidence_threshold: float = 0.5,
    label_map: Mapping[str, str | Emotion] | None = None,
    noise_snr_db: Sequence[float] = (20.0, 10.0, 5.0, 0.0),
    interference_sir_db: Sequence[float] = (-6.0, 0.0, 6.0),
) -> dict:
    manifest = load_manifest(manifest_path, label_map=label_map)
    if not confirm_real_labeled_data:
        return {
            "status": "NOT_EVALUATED_NO_LABELED_DATA",
            "real_performance_evaluated": False,
            "identity_status": "NOT_EVALUATED",
            "manifest_record_count": len(manifest.records),
            "reason": "Pass the explicit confirmation only for licensed real labeled recordings; synthetic data is program-test input, not performance evidence.",
        }

    predictor = SklearnEmotionPredictor.load(model_path)
    if predictor.model_status is not ModelStatus.READY:
        return {
            "status": "MODEL_NOT_READY",
            "model_status": predictor.model_status.value,
            "real_performance_evaluated": False,
            "identity_status": "NOT_EVALUATED",
        }
    metadata = predictor.metadata
    split_metadata = metadata.get("dataset_split", {})
    saved_label_map = split_metadata.get("label_mapping")
    if saved_label_map and label_map is None:
        manifest = load_manifest(manifest_path, label_map=saved_label_map)
    if saved_label_map and manifest.label_mapping != saved_label_map:
        return {"status": "LABEL_MAPPING_MISMATCH", "real_performance_evaluated": False, "identity_status": "NOT_EVALUATED"}
    selected_seed = int(seed if seed is not None else split_metadata.get("seed", 42))
    manifest_sha = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
    if split_metadata.get("manifest_sha256") and split_metadata["manifest_sha256"] != manifest_sha:
        return {
            "status": "MANIFEST_MISMATCH",
            "real_performance_evaluated": False,
            "identity_status": "NOT_EVALUATED",
        }
    label_sha = hashlib.sha256(json.dumps(manifest.label_mapping, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    if split_metadata.get("label_mapping_sha256") and split_metadata["label_mapping_sha256"] != label_sha:
        return {"status": "LABEL_MAPPING_MISMATCH", "real_performance_evaluated": False, "identity_status": "NOT_EVALUATED"}
    if int(metadata.get("sample_rate", 0)) <= 0:
        return {"status": "MODEL_METADATA_INVALID", "real_performance_evaluated": False, "identity_status": "NOT_EVALUATED"}

    split = split_by_speaker_and_recording(manifest.records, seed=selected_seed)
    if split_metadata.get("record_counts", {}).get("test") not in (None, len(split.test)):
        return {"status": "SPLIT_MISMATCH", "real_performance_evaluated": False, "identity_status": "NOT_EVALUATED"}

    classes = tuple(metadata["classes"])
    conditions = [("clean", None, None)]
    conditions.extend((f"white_noise_{snr:g}db", float(snr), None) for snr in noise_snr_db)
    conditions.extend((f"other_voice_sir_{sir:g}db", None, float(sir)) for sir in interference_sir_db)
    reports = {}
    feature_times: list[float] = []
    prediction_times: list[float] = []
    rss_sampler = _PeakRssSampler()
    rss_sampler.start()
    try:
        audio_cache: dict[Path, AudioBuffer] = {}
        for condition_index, (condition_name, snr_db, sir_db) in enumerate(conditions):
            labels: list[str] = []
            probabilities: list[list[float]] = []
            eligible: list[bool] = []
            reject_reasons: Counter[str] = Counter()
            for row_index, record in enumerate(split.test):
                labels.append(record.emotion.value)
                audio = load_wav(record.path)
                if audio.sample_rate != metadata["sample_rate"]:
                    probabilities.append(_uniform_probabilities(len(classes)))
                    eligible.append(False)
                    reject_reasons["SAMPLE_RATE_MISMATCH"] += 1
                    continue
                samples = audio.samples
                try:
                    if snr_db is not None:
                        samples = add_white_noise_snr(samples, snr_db, seed=selected_seed + row_index)
                    elif sir_db is not None:
                        other_voice = _find_other_speaker_record(split.test, record)
                        if other_voice is None:
                            raise ValueError("no other-speaker test recording is available")
                        other_audio = audio_cache.get(other_voice.path)
                        if other_audio is None:
                            other_audio = load_wav(other_voice.path)
                            audio_cache[other_voice.path] = other_audio
                        if other_audio.sample_rate != audio.sample_rate:
                            raise ValueError("interferer sample rate mismatch")
                        samples = add_interferer_sir(samples, other_audio.samples, sir_db)
                except ValueError as error:
                    probabilities.append(_uniform_probabilities(len(classes)))
                    eligible.append(False)
                    reason = "NO_INTERFERER" if "other-speaker" in str(error) else "MIX_INVALID"
                    reject_reasons[reason] += 1
                    continue

                test_audio = AudioBuffer(samples, audio.sample_rate)
                quality = extract_features(
                    test_audio,
                    frame_ms=metadata["frame_ms"],
                    activity_rms_threshold=metadata["activity_rms_threshold"],
                )
                if quality.quality_status is not AudioQualityStatus.ACCEPTABLE:
                    probabilities.append(_uniform_probabilities(len(classes)))
                    eligible.append(False)
                    reject_reasons[quality.quality_status.value] += 1
                    continue

                started = time.perf_counter_ns()
                prosody = extract_prosody(
                    test_audio,
                    frame_ms=metadata["frame_ms"],
                    hop_ms=metadata["hop_ms"],
                    activity_rms_threshold=metadata["activity_rms_threshold"],
                )
                feature_times.append((time.perf_counter_ns() - started) / 1e6)
                started = time.perf_counter_ns()
                prediction = predictor.predict(prosody)
                prediction_times.append((time.perf_counter_ns() - started) / 1e6)
                if prediction.model_status is not ModelStatus.READY or prediction.emotion is None:
                    probabilities.append(_uniform_probabilities(len(classes)))
                    eligible.append(False)
                    reject_reasons[prediction.status.value] += 1
                    continue
                probabilities.append([prediction.scores.get(Emotion(label), 0.0) for label in classes])
                eligible.append(True)

            metrics = compute_classification_metrics(
                labels,
                np.asarray(probabilities, dtype=np.float64),
                classes,
                eligible=eligible,
                confidence_threshold=confidence_threshold,
            )
            reports[condition_name] = {
                "snr_db": snr_db,
                "other_voice_sir_db": sir_db,
                "metrics": metrics,
                "rejection_reasons": dict(reject_reasons),
            }
    finally:
        rss_sampler.stop()

    return {
        "status": "EVALUATED_USER_ATTESTED_LABELED_DATA",
        "real_performance_evaluated": True,
        "identity_status": "NOT_EVALUATED",
        "interpretation": "Emotion-only offline evaluation; not target-speaker verification or a clinical probability.",
        "model_version": metadata["model_version"],
        "model_size_bytes": predictor.model_size_bytes,
        "manifest_sha256": manifest_sha,
        "seed": selected_seed,
        "test_record_count": len(split.test),
        "feature_extraction_ms": _latency_summary(feature_times),
        "predictor_inference_ms": _latency_summary(prediction_times),
        "peak_rss_bytes": int(rss_sampler.peak_bytes),
        "peak_rss_measurement": "psutil RSS sampled every 20 ms; process-wide peak during this evaluation",
        "conditions": reports,
    }


def _find_other_speaker_record(records, target_record):
    return next((
        candidate for candidate in records
        if candidate.speaker_id != target_record.speaker_id
        and (candidate.dataset_id, candidate.recording_id) != (target_record.dataset_id, target_record.recording_id)
    ), None)


def _uniform_probabilities(class_count: int) -> list[float]:
    return [1.0 / class_count] * class_count


def _latency_summary(values: list[float]) -> dict:
    if not values:
        return {"sample_count": 0, "median": None, "p95": None}
    return {
        "sample_count": len(values),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="emolight-evaluate", description="Evaluate an emotion-only SVM on a speaker-exclusive local test split")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--label-map", help="JSON object mapping corpus labels to neutral/happy/angry/sad")
    parser.add_argument("--confirm-real-labeled-data", action="store_true", help="attest the manifest is licensed real labeled data; without this no performance metrics are emitted")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    args = parser.parse_args(argv)
    try:
        label_map = json.loads(Path(args.label_map).read_text(encoding="utf-8")) if args.label_map else None
        if label_map is not None and not isinstance(label_map, dict):
            raise ValueError("label map JSON must be an object")
        result = evaluate_manifest(
            args.manifest,
            args.model,
            confirm_real_labeled_data=args.confirm_real_labeled_data,
            seed=args.seed,
            confidence_threshold=args.confidence_threshold,
            label_map=label_map,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"emolight-evaluate: {error}\n")
    output = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    else:
        print(output)
    return 0 if result.get("status") in ("NOT_EVALUATED_NO_LABELED_DATA", "EVALUATED_USER_ATTESTED_LABELED_DATA") else 2
