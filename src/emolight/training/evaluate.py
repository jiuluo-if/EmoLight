"""Held-out evaluation for validation-calibrated, NumPy-exportable linear models."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Mapping, Sequence

import numpy as np

from emolight.audio.wav import AudioBuffer, load_wav
from emolight.data.manifest import load_manifest
from emolight.data.split import split_by_speaker_and_recording
from emolight.emotion.linear import NumpyLinearEmotionPredictor
from emolight.emotion.predictor import ModelStatus
from emolight.events import Emotion
from emolight.features.acoustic import AudioQualityStatus, extract_features
from emolight.features.prosody import extract_prosody
from emolight.training.noise import (
    add_background_noise_snr,
    add_interferer_sir,
    add_white_noise_snr,
    apply_reverb,
    load_noise_manifest,
)


def compute_classification_metrics(
    true_labels: Sequence[str],
    probabilities: np.ndarray,
    classes: Sequence[str],
    *,
    eligible: Sequence[bool] | None = None,
    confidence_threshold: float = 0.5,
) -> dict:
    """Compute selective classification metrics without hiding unsupported labels."""
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

    confidence = np.max(probs, axis=1)
    predictions = np.asarray(class_order)[np.argmax(probs, axis=1)]
    accepted = valid & (confidence >= confidence_threshold)
    accepted_labels = y_true[accepted]
    accepted_predictions = predictions[accepted]
    accepted_errors = int(np.sum(accepted_labels != accepted_predictions))
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
                "support": int(support[index]),
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
        "coverage": float(accepted.mean()),
        "rejection_rate": float(1.0 - accepted.mean()),
        "accepted_error_count": accepted_errors,
        "accepted_error_rate": float(accepted_errors / accepted.sum()) if accepted.any() else None,
        "confidence_threshold": confidence_threshold,
        "macro_f1": macro_f1,
        "uar": uar,
        "per_class": per_class,
        "confusion_matrix_labels": list(report_classes),
        "confusion_matrix": confusion.astype(int).tolist(),
        "calibration": calibration,
        "unsupported_truth_count": int(np.sum(~np.isin(y_true, class_order))),
    }


def _aggregate_recordings(
    *,
    labels: Sequence[str],
    probabilities: np.ndarray,
    eligible: Sequence[bool],
    speaker_ids: Sequence[str],
    recording_keys: Sequence[object],
) -> dict:
    """Average usable window scores by original recording, keeping all-rejected recordings rejected."""
    labels_array = np.asarray(labels, dtype=str)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    eligible_array = np.asarray(eligible, dtype=bool)
    speakers = tuple(str(value) for value in speaker_ids)
    keys = tuple(recording_keys)
    count = labels_array.size
    if probabilities.ndim != 2 or probabilities.shape[0] != count or any(
        len(values) != count for values in (eligible_array, speakers, keys)
    ):
        raise ValueError("recording aggregation inputs must have one row per window")
    grouped: dict[object, list[int]] = {}
    for index, key in enumerate(keys):
        grouped.setdefault(key, []).append(index)
    aggregated_labels = []
    aggregated_probabilities = []
    aggregated_eligible = []
    aggregated_speakers = []
    for key, indices in grouped.items():
        group_labels = {labels_array[index] for index in indices}
        group_speakers = {speakers[index] for index in indices}
        if len(group_labels) != 1 or len(group_speakers) != 1:
            raise ValueError("one source recording must have one label and one speaker")
        valid_indices = [index for index in indices if eligible_array[index]]
        aggregated_labels.append(next(iter(group_labels)))
        aggregated_speakers.append(next(iter(group_speakers)))
        aggregated_eligible.append(bool(valid_indices))
        if valid_indices:
            aggregated_probabilities.append(np.mean(probabilities[valid_indices], axis=0))
        else:
            aggregated_probabilities.append(np.full(probabilities.shape[1], 1.0 / probabilities.shape[1]))
    return {
        "labels": tuple(aggregated_labels),
        "probabilities": np.asarray(aggregated_probabilities, dtype=np.float64),
        "eligible": tuple(aggregated_eligible),
        "speaker_ids": tuple(aggregated_speakers),
    }


def _recording_bootstrap_intervals(
    aggregate: dict,
    classes: Sequence[str],
    threshold: float,
    *,
    seed: int,
    resamples: int = 1000,
) -> dict:
    labels = np.asarray(aggregate["labels"], dtype=str)
    probabilities = aggregate["probabilities"]
    eligible = np.asarray(aggregate["eligible"], dtype=bool)
    strata = [np.flatnonzero(labels == label) for label in sorted(set(labels))]
    metric_names = ("macro_f1", "uar", "coverage", "accepted_error_rate")
    recall_names = tuple(f"recall_{label}" for label in classes)
    samples: dict[str, list[float]] = {key: [] for key in (*metric_names, *recall_names)}
    generator = np.random.default_rng(seed)
    for _ in range(resamples):
        indices = np.concatenate([generator.choice(group, size=len(group), replace=True) for group in strata])
        metrics = compute_classification_metrics(
            labels[indices], probabilities[indices], classes,
            eligible=eligible[indices], confidence_threshold=threshold,
        )
        for key in metric_names:
            value = metrics[key]
            if value is not None:
                samples[key].append(float(value))
        for label, key in zip(classes, recall_names):
            samples[key].append(float(metrics["per_class"].get(label, {}).get("recall", 0.0)))
    return {
        "unit": "original recording, stratified by true class",
        "method": "percentile bootstrap with replacement",
        "resamples": resamples,
        "confidence_level": 0.95,
        "seed": seed,
        "intervals": {
            key: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))] if values else None
            for key, values in samples.items()
        },
    }


def _summarize_per_speaker(results: Mapping[str, dict]) -> dict:
    """Publish cross-speaker metric ranges without exposing dataset speaker keys."""
    metric_fields = {
        "macro_f1": lambda value: value.get("macro_f1"),
        "uar": lambda value: value.get("uar"),
        "happy_recall": lambda value: value.get("per_class", {}).get("happy", {}).get("recall"),
        "coverage": lambda value: value.get("coverage"),
        "accepted_error_rate": lambda value: value.get("accepted_error_rate"),
    }
    values = list(results.values())
    ranges = {}
    for name, extract in metric_fields.items():
        observed = [float(value) for result in values if (value := extract(result)) is not None]
        ranges[name] = [min(observed), max(observed)] if observed else None
    recording_counts = [int(result["sample_count"]) for result in values]
    return {
        "speaker_count": len(values),
        "recordings_per_speaker": [min(recording_counts), max(recording_counts)] if recording_counts else None,
        "metric_ranges": ranges,
    }


def _without_public_speaker_identifiers(value):
    if isinstance(value, list):
        return [_without_public_speaker_identifiers(item) for item in value]
    if not isinstance(value, dict):
        return value
    private_keys = {"speaker_id", "speaker_ids", "speaker_name", "speaker_names", "split_speaker_ids", "speaker_ids_by_split", "test_speaker_ids"}
    return {
        key: _without_public_speaker_identifiers(item)
        for key, item in value.items()
        if key not in private_keys
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
    ece = 0.0
    bins = np.linspace(0.0, 1.0, 11)
    for index in range(len(bins) - 1):
        in_bin = (confidence >= bins[index]) & (
            confidence < bins[index + 1] if index < len(bins) - 2 else confidence <= bins[index + 1]
        )
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
    label_map: Mapping[str, str | Emotion] | None = None,
    noise_manifest_path: str | Path | None = None,
    noise_snr_db: Sequence[float] = (20.0, 10.0, 5.0, 0.0),
    interference_sir_db: Sequence[float] = (6.0, 0.0, -6.0),
) -> dict:
    manifest = load_manifest(manifest_path, label_map=label_map)
    if not confirm_real_labeled_data:
        return {
            "status": "NOT_EVALUATED_NO_LABELED_DATA",
            "real_performance_evaluated": False,
            "identity_status": "NOT_EVALUATED",
            "manifest_record_count": len(manifest.records),
            "reason": "Pass the explicit confirmation only for licensed real labeled recordings; generated test audio is not performance evidence.",
        }

    predictor = NumpyLinearEmotionPredictor.load(model_path)
    if predictor.model_status is not ModelStatus.READY:
        return {
            "status": "MODEL_NOT_READY",
            "model_status": predictor.model_status.value,
            "real_performance_evaluated": False,
            "identity_status": "NOT_EVALUATED",
        }
    model = predictor.metadata
    training_metadata = model.get("training", {})
    saved_label_mapping = model.get("label_mapping")
    if saved_label_mapping and label_map is None:
        manifest = load_manifest(manifest_path, label_map=saved_label_mapping)
    if saved_label_mapping and manifest.label_mapping != saved_label_mapping:
        return _not_evaluated("LABEL_MAPPING_MISMATCH")
    manifest_hash = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
    if model.get("training_manifest_sha256") != manifest_hash:
        return _not_evaluated("MANIFEST_MISMATCH")
    selected_seed = int(model.get("dataset_split", {}).get("seed", 42))
    if seed is not None and seed != selected_seed:
        return _not_evaluated("SPLIT_MISMATCH", requested_seed=seed, trained_seed=selected_seed)
    split = split_by_speaker_and_recording(manifest.records, seed=selected_seed)
    if split.to_metadata(include_test_labels=False) != model.get("dataset_split"):
        return _not_evaluated("SPLIT_MISMATCH")

    classes = tuple(model["classes"])
    validation = model["validation"]
    rejection_threshold = float(model["rejection_threshold"])
    quality_settings = model["quality_gates"]
    noise_sources, noise_metadata = load_noise_manifest(noise_manifest_path) if noise_manifest_path else ({}, {})
    sampler = _PeakRssSampler()
    import psutil

    process = psutil.Process()
    cpu_before = process.cpu_times()
    wall_started = time.perf_counter_ns()
    load_times: list[float] = []
    vad_times: list[float] = []
    feature_times: list[float] = []
    inference_times: list[float] = []
    total_times: list[float] = []
    sampler.start()
    conditions: dict[str, dict] = {}
    try:
        conditions["clean"] = _evaluate_condition(
            "clean", split.test, classes, predictor, rejection_threshold, model,
            lambda record, audio, index: audio.samples,
            load_times, vad_times, feature_times, inference_times, total_times,
            recording_summary=True,
        )
        for snr_db in noise_snr_db:
            condition_name = f"white_noise_{snr_db:g}db"
            conditions[condition_name] = _evaluate_condition(
                condition_name, split.test, classes, predictor, rejection_threshold, model,
                lambda record, audio, index, snr=float(snr_db): add_white_noise_snr(audio.samples, snr, seed=selected_seed + index),
                load_times, vad_times, feature_times, inference_times, total_times,
                snr_db=float(snr_db),
            )
        for noise_type in ("background_music", "fan", "environment"):
            sources = noise_sources.get(noise_type, ())
            if not sources:
                conditions[noise_type] = {"status": "NOT_EVALUATED_NO_SOURCE", "metrics": None}
                continue
            for snr_db in noise_snr_db:
                condition_name = f"{noise_type}_snr_{snr_db:g}db"
                conditions[condition_name] = _evaluate_condition(
                    condition_name, split.test, classes, predictor, rejection_threshold, model,
                    lambda record, audio, index, snr=float(snr_db), values=sources: _mix_background_source(
                        audio, values[index % len(values)], snr
                    ),
                    load_times, vad_times, feature_times, inference_times, total_times,
                    snr_db=float(snr_db), source_count=len(sources),
                )
        rir_sources = noise_sources.get("reverb_impulse", ())
        if not rir_sources:
            conditions["reverb"] = {"status": "NOT_EVALUATED_NO_RIR", "metrics": None}
        else:
            for source_index, rir in enumerate(rir_sources):
                name = f"reverb_rir_{source_index + 1}"
                conditions[name] = _evaluate_condition(
                    name, split.test, classes, predictor, rejection_threshold, model,
                    lambda record, audio, index, impulse=rir: _apply_rir(audio, impulse),
                    load_times, vad_times, feature_times, inference_times, total_times,
                    source_count=1,
                )
        overlap_exists = len({(r.dataset_id, r.speaker_id) for r in split.test}) >= 2
        if not overlap_exists:
            conditions["other_speaker_overlap"] = {"status": "NOT_EVALUATED_NO_OTHER_SPEAKER", "metrics": None}
        else:
            for sir_db in interference_sir_db:
                condition_name = f"other_voice_sir_{sir_db:g}db"
                conditions[condition_name] = _evaluate_condition(
                    condition_name, split.test, classes, predictor, rejection_threshold, model,
                    lambda record, audio, index, sir=float(sir_db): _mix_other_speaker(
                        audio, _find_other_speaker_record(split.test, record), sir
                    ),
                    load_times, vad_times, feature_times, inference_times, total_times,
                    other_voice_sir_db=float(sir_db),
                )
    finally:
        sampler.stop()
    cpu_after = process.cpu_times()
    wall_seconds = (time.perf_counter_ns() - wall_started) / 1e9
    cpu_seconds = (cpu_after.user - cpu_before.user) + (cpu_after.system - cpu_before.system)
    test_split_summary = split.to_metadata()["subsets"]["test"]
    dataset_provenance = model.get("dataset_provenance", {})
    dataset_provenance = _without_public_speaker_identifiers(dataset_provenance)
    parameter_count = _model_parameter_count(model)
    return {
        "status": "EVALUATED_USER_ATTESTED_LABELED_DATA",
        "real_performance_evaluated": True,
        "data_attestation": "User confirmed licensed real labeled recordings.",
        "identity_status": "NOT_EVALUATED",
        "interpretation": "Emotion-only held-out evaluation; no speaker verification or clinical inference.",
        "model_version": model["model_version"],
        "model_profile": model["feature_profile"],
        "dataset_provenance": dataset_provenance,
        "inference_window_s": model.get("window_s"),
        "inference_update_interval_s": model.get("update_interval_s"),
        "evaluation_unit": "manifest row; this dataset contains overlapping windows, so rows from one utterance are correlated",
        "model_size_bytes": predictor.model_size_bytes,
        "model_parameter_count": parameter_count,
        "model_validation": {
            "calibration_method": model["calibration_method"],
            "threshold_selection_method": validation["threshold_selection_method"],
            "threshold": rejection_threshold,
            "validation_brier_score": validation["brier_score"],
            "validation_ece": validation["ece"],
            "validation_manifest_sha256": validation["manifest_sha256"],
        },
        "manifest_sha256": manifest_hash,
        "seed": selected_seed,
        "test_record_count": len(split.test),
        "test_unique_recording_count": len({(record.dataset_id, record.recording_id) for record in split.test}),
        "test_speaker_count": len({record.speaker_id for record in split.test}),
        "test_class_counts": test_split_summary["class_counts"],
        "test_missing_classes": test_split_summary["missing_classes"],
        "latency_ms": {
            "wav_load": _latency_summary(load_times),
            "quality_and_vad": _latency_summary(vad_times),
            "prosody_feature_extraction": _latency_summary(feature_times),
            "numpy_model_inference": _latency_summary(inference_times),
            "end_to_end_per_test_item": _latency_summary(total_times),
        },
        "resource_usage": {
            "process_cpu_seconds": cpu_seconds,
            "wall_seconds": wall_seconds,
            "average_cpu_cores_used": cpu_seconds / wall_seconds if wall_seconds else None,
            "peak_rss_bytes": int(sampler.peak_bytes),
        },
        "conditions": conditions,
        "noise_sources": noise_metadata,
        "noise_source_counts": {kind: len(entries) for kind, entries in noise_sources.items()},
    }


def evaluate_model_pair(
    manifest_path: str | Path,
    simple_model_path: str | Path,
    full_model_path: str | Path,
    **evaluation_options,
) -> dict:
    """Evaluate both profiles against the exact same held-out manifest and settings."""
    simple = evaluate_manifest(manifest_path, simple_model_path, **evaluation_options)
    full = evaluate_manifest(manifest_path, full_model_path, **evaluation_options)
    comparisons: dict[str, dict] = {}
    for condition in sorted(set(simple.get("conditions", {})) & set(full.get("conditions", {}))):
        simple_metrics = (simple["conditions"][condition].get("metrics") or {})
        full_metrics = (full["conditions"][condition].get("metrics") or {})
        deltas = {}
        for metric in ("macro_f1", "uar", "coverage", "rejection_rate", "accepted_error_rate"):
            left, right = simple_metrics.get(metric), full_metrics.get(metric)
            if left is not None and right is not None:
                deltas[metric] = float(right - left)
        comparisons[condition] = {
            "simple_status": simple["conditions"][condition].get("status", "EVALUATED"),
            "full_status": full["conditions"][condition].get("status", "EVALUATED"),
            "simple_metrics": simple_metrics or None,
            "full_metrics": full_metrics or None,
            "delta_full_minus_simple": deltas or None,
        }
    return {
        "status": "PAIRED_EVALUATION",
        "comparison_protocol": {
            "same_manifest": True,
            "same_seed_and_conditions": True,
            "metrics_are_test_set_only": True,
            "delta_direction": "full minus simple; positive error-rate delta is worse",
        },
        "simple": simple,
        "full": full,
        "conditions": comparisons,
    }


def _evaluate_condition(
    name: str,
    records,
    classes: tuple[str, ...],
    predictor: NumpyLinearEmotionPredictor,
    threshold: float,
    model: dict,
    variant,
    load_times: list[float],
    vad_times: list[float],
    feature_times: list[float],
    inference_times: list[float],
    total_times: list[float],
    recording_summary: bool = False,
    **metadata,
) -> dict:
    labels: list[str] = []
    probabilities: list[list[float]] = []
    eligible: list[bool] = []
    speaker_ids: list[str] = []
    recording_keys: list[tuple[str, str]] = []
    rejects: Counter[str] = Counter()
    for index, record in enumerate(records):
        total_started = time.perf_counter_ns()
        labels.append(record.emotion.value)
        speaker_ids.append(record.speaker_id)
        recording_keys.append((record.dataset_id, record.recording_id))
        load_started = time.perf_counter_ns()
        audio = load_wav(record.path)
        load_times.append((time.perf_counter_ns() - load_started) / 1e6)
        if audio.sample_rate != model["sample_rate"]:
            probabilities.append(_uniform_probabilities(len(classes)))
            eligible.append(False)
            rejects["SAMPLE_RATE_MISMATCH"] += 1
            total_times.append((time.perf_counter_ns() - total_started) / 1e6)
            continue
        try:
            samples = variant(record, audio, index)
        except (ValueError, OSError) as error:
            probabilities.append(_uniform_probabilities(len(classes)))
            eligible.append(False)
            rejects["MIX_INVALID"] += 1
            total_times.append((time.perf_counter_ns() - total_started) / 1e6)
            continue
        mixed_audio = AudioBuffer(samples, audio.sample_rate)
        gates = model["quality_gates"]
        vad_started = time.perf_counter_ns()
        quality = extract_features(
            mixed_audio,
            frame_ms=model["frame_ms"],
            activity_rms_threshold=model["activity_rms_threshold"],
            min_activity_ratio=gates["min_activity_ratio"],
            min_contiguous_active_frames=gates["min_contiguous_active_frames"],
            clipping_limit=gates["clipping_limit"],
            minimum_snr_db=gates["minimum_snr_db"],
            acceptable_snr_db=gates["acceptable_snr_db"],
        )
        vad_times.append((time.perf_counter_ns() - vad_started) / 1e6)
        if quality.quality_status is not AudioQualityStatus.ACCEPTABLE:
            probabilities.append(_uniform_probabilities(len(classes)))
            eligible.append(False)
            rejects[quality.quality_status.value] += 1
            total_times.append((time.perf_counter_ns() - total_started) / 1e6)
            continue

        feature_started = time.perf_counter_ns()
        features = extract_prosody(
            mixed_audio,
            frame_ms=model["frame_ms"],
            hop_ms=model["hop_ms"],
            activity_rms_threshold=model["activity_rms_threshold"],
        )
        feature_times.append((time.perf_counter_ns() - feature_started) / 1e6)
        if features.valid_frame_fraction < 0.95:
            probabilities.append(_uniform_probabilities(len(classes)))
            eligible.append(False)
            rejects["INVALID_FRAME_CONTENT"] += 1
            total_times.append((time.perf_counter_ns() - total_started) / 1e6)
            continue
        inference_started = time.perf_counter_ns()
        prediction = predictor.predict(features)
        inference_times.append((time.perf_counter_ns() - inference_started) / 1e6)
        if prediction.model_status is not ModelStatus.READY or not prediction.scores:
            probabilities.append(_uniform_probabilities(len(classes)))
            eligible.append(False)
            rejects[prediction.model_status.value] += 1
        else:
            probabilities.append([prediction.scores.get(Emotion(label), 0.0) for label in classes])
            eligible.append(True)
            if prediction.emotion is None:
                rejects["BELOW_VALIDATION_THRESHOLD"] += 1
        total_times.append((time.perf_counter_ns() - total_started) / 1e6)

    metrics = compute_classification_metrics(
        labels,
        np.asarray(probabilities, dtype=np.float64),
        classes,
        eligible=eligible,
        confidence_threshold=threshold,
    )
    result = {"status": "EVALUATED", "metadata": metadata, "metrics": metrics, "rejection_reasons": dict(rejects)}
    if recording_summary and labels:
        aggregate = _aggregate_recordings(
            labels=labels,
            probabilities=np.asarray(probabilities, dtype=np.float64),
            eligible=eligible,
            speaker_ids=speaker_ids,
            recording_keys=recording_keys,
        )
        result["recording_aggregated"] = compute_classification_metrics(
            aggregate["labels"], aggregate["probabilities"], classes,
            eligible=aggregate["eligible"], confidence_threshold=threshold,
        )
        result["recording_aggregation_method"] = "mean calibrated class scores over quality-eligible overlapping windows; reject if every window is ineligible"
        result["recording_bootstrap_95ci"] = _recording_bootstrap_intervals(
            aggregate, classes, threshold, seed=int(model.get("dataset_split", {}).get("seed", 42))
        )
        per_speaker = {}
        aggregate_speakers = np.asarray(aggregate["speaker_ids"], dtype=str)
        for speaker_id in sorted(set(aggregate["speaker_ids"])):
            indices = np.flatnonzero(aggregate_speakers == speaker_id)
            per_speaker[speaker_id] = compute_classification_metrics(
                np.asarray(aggregate["labels"])[indices],
                aggregate["probabilities"][indices],
                classes,
                eligible=np.asarray(aggregate["eligible"], dtype=bool)[indices],
                confidence_threshold=threshold,
            )
        result["per_speaker_summary"] = _summarize_per_speaker(per_speaker)
    return result


def _mix_background_source(target: AudioBuffer, background: AudioBuffer, snr_db: float) -> np.ndarray:
    if target.sample_rate != background.sample_rate:
        raise ValueError("background noise sample rate mismatch")
    return add_background_noise_snr(target.samples, background.samples, snr_db)


def _apply_rir(target: AudioBuffer, rir: AudioBuffer) -> np.ndarray:
    if target.sample_rate != rir.sample_rate:
        raise ValueError("room impulse response sample rate mismatch")
    return apply_reverb(target.samples, rir.samples)


def _mix_other_speaker(target: AudioBuffer, other_record, sir_db: float) -> np.ndarray:
    if other_record is None:
        raise ValueError("no other-speaker test recording is available")
    other = load_wav(other_record.path)
    if target.sample_rate != other.sample_rate:
        raise ValueError("other-speaker sample rate mismatch")
    return add_interferer_sir(target.samples, other.samples, sir_db)


def _find_other_speaker_record(records, target_record):
    return next((
        candidate for candidate in records
        if (candidate.dataset_id, candidate.speaker_id) != (target_record.dataset_id, target_record.speaker_id)
        and (candidate.dataset_id, candidate.recording_id) != (target_record.dataset_id, target_record.recording_id)
    ), None)


def _uniform_probabilities(class_count: int) -> list[float]:
    return [1.0 / class_count] * class_count


def _model_parameter_count(model: dict) -> int:
    return int(sum(np.asarray(model[key]).size for key in (
        "weights", "intercepts", "standardization_mean", "standardization_scale", "imputation_values",
        "calibration_slopes", "calibration_intercepts",
    )))


def _latency_summary(values: list[float]) -> dict:
    if not values:
        return {"sample_count": 0, "median": None, "p95": None}
    return {
        "sample_count": len(values),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
    }


def _not_evaluated(status: str, **extra) -> dict:
    return {
        "status": status,
        "real_performance_evaluated": False,
        "identity_status": "NOT_EVALUATED",
        **extra,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evaluate_linear.py", description="Evaluate a NumPy linear model on held-out speaker-exclusive recordings")
    parser.add_argument("--manifest", required=True)
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--model", help="evaluate one model")
    model_group.add_argument("--simple-model", help="evaluate simple and full models as a paired comparison")
    parser.add_argument("--full-model", help="full model for --simple-model paired comparison")
    parser.add_argument("--noise-manifest", help="CSV noise_type,path for recorded music/fan/RIR sources")
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--label-map")
    parser.add_argument("--confirm-real-labeled-data", action="store_true", help="confirm the labeled recordings are real and licensed")
    args = parser.parse_args(argv)
    try:
        if not args.model and not (args.simple_model and args.full_model):
            parser.error("provide --model or both --simple-model and --full-model")
        if args.model and args.full_model:
            parser.error("--full-model is only used with --simple-model")
        label_map = json.loads(Path(args.label_map).read_text(encoding="utf-8")) if args.label_map else None
        if label_map is not None and not isinstance(label_map, dict):
            raise ValueError("label map JSON must be an object")
        options = {
            "confirm_real_labeled_data": args.confirm_real_labeled_data,
            "seed": args.seed,
            "label_map": label_map,
        }
        if args.model:
            result = evaluate_manifest(args.manifest, args.model, **options)
        else:
            result = evaluate_model_pair(args.manifest, args.simple_model, args.full_model, **options)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"evaluate_linear.py: operation failed ({type(error).__name__})\n")
    output = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    else:
        print(output)
    return 0 if result.get("status") in (
        "NOT_EVALUATED_NO_LABELED_DATA", "EVALUATED_USER_ATTESTED_LABELED_DATA", "MODEL_NOT_READY", "PAIRED_EVALUATION",
    ) else 2
