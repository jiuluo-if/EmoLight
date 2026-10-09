"""Train simple and full linear baselines; deploy artifacts contain JSON only."""

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import numpy as np

from emolight.audio.wav import AudioBuffer, load_wav
from emolight.data.manifest import load_manifest
from emolight.data.split import DatasetSplit, split_by_speaker_and_recording
from emolight.emotion.linear import MODEL_FORMAT_VERSION, MODEL_VERSION
from emolight.events import Emotion
from emolight.features.acoustic import AudioQualityStatus, extract_features
from emolight.features.emotion_v1 import (
    EMOTION_FEATURE_NAMES,
    EMOTION_FEATURE_SCHEMA_VERSION,
    SIMPLE_FEATURE_NAMES,
    SIMPLE_FEATURE_SCHEMA_VERSION,
    EmotionFeatures,
    extract_emotion_features,
)
from emolight.training.noise import (
    add_background_noise_snr,
    add_white_noise_snr,
    apply_reverb,
    load_noise_manifest,
)


_CLASS_NAMES = tuple(emotion.value for emotion in Emotion)


def train_linear_models(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 42,
    sample_rate: int = 16000,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    activity_rms_threshold: float = 0.01,
    white_noise_augmentation_snr_db: Sequence[float] = (),
    background_noise_manifest_path: str | Path | None = None,
    background_noise_augmentation_snr_db: Sequence[float] = (),
    augment_reverb: bool = False,
    label_map: Mapping[str, str | Emotion] | None = None,
) -> dict:
    """Fit simple/full models, calibrate only on validation, and leave test untouched."""
    manifest = load_manifest(manifest_path, label_map=label_map)
    if not manifest.records:
        raise ValueError("manifest has no supported labeled audio rows")
    split = split_by_speaker_and_recording(manifest.records, seed=seed)
    split_metadata = split.to_metadata()
    all_classes = {record.emotion.value for record in manifest.records}
    train_classes = {record.emotion.value for record in split.train}
    if train_classes != all_classes:
        missing = sorted(all_classes - train_classes)
        raise ValueError(f"training split is missing emotion classes: {', '.join(missing)}; split={split_metadata['subsets']}")

    records = {name: subset for name, subset in (
        ("train", split.train),
        ("validation", split.validation),
    )}
    features: dict[str, list[EmotionFeatures]] = {}
    accepted_records: dict[str, list] = {}
    rejects: dict[str, Counter[str]] = {}
    for name, subset in records.items():
        features[name], accepted_records[name], rejects[name] = _extract_acceptable_features(
            subset,
            sample_rate=sample_rate,
            frame_ms=frame_ms,
            hop_ms=hop_ms,
            activity_rms_threshold=activity_rms_threshold,
        )
    if not features["train"]:
        raise ValueError(f"no ACCEPTABLE training windows remain after quality gating: {dict(rejects['train'])}")
    if not features["validation"]:
        raise ValueError(f"no ACCEPTABLE validation windows remain after quality gating: {dict(rejects['validation'])}")
    train_labels = [record.emotion.value for record in accepted_records["train"]]
    augmentation_counts: dict[str, int] = {}
    for snr_index, snr_db in enumerate(white_noise_augmentation_snr_db):
        if not np.isfinite(snr_db) or not -20.0 <= snr_db <= 60.0:
            raise ValueError("white noise augmentation SNR must be within [-20, 60] dB")
        added = 0
        for row_index, record in enumerate(accepted_records["train"]):
            audio = load_wav(record.path)
            augmented_samples = add_white_noise_snr(
                audio.samples,
                float(snr_db),
                seed=seed + 1009 * (snr_index + 1) + row_index,
            )
            augmented_features = extract_emotion_features(
                AudioBuffer(augmented_samples, audio.sample_rate),
                frame_ms=frame_ms,
                hop_ms=hop_ms,
                activity_rms_threshold=activity_rms_threshold,
            )
            if augmented_features.valid_frame_fraction < 0.95:
                rejects["train"]["AUGMENT_INVALID_FRAME_CONTENT"] += 1
                continue
            features["train"].append(augmented_features)
            train_labels.append(record.emotion.value)
            added += 1
        augmentation_counts[f"{snr_db:g}dB"] = added

    noise_sources, noise_metadata = load_noise_manifest(background_noise_manifest_path) if background_noise_manifest_path else ({}, {})
    if background_noise_augmentation_snr_db and not any(
        noise_sources.get(kind) for kind in ("background_music", "fan", "environment")
    ):
        raise ValueError("recorded noise augmentation requested but no music/fan/environment source is available")
    for noise_type in ("background_music", "fan", "environment"):
        sources = noise_sources.get(noise_type, ())
        if not sources:
            continue
        for snr_index, snr_db in enumerate(background_noise_augmentation_snr_db):
            if not np.isfinite(snr_db) or not -20.0 <= snr_db <= 60.0:
                raise ValueError("background noise augmentation SNR must be within [-20, 60] dB")
            added = 0
            for row_index, record in enumerate(accepted_records["train"]):
                audio = load_wav(record.path)
                source = sources[row_index % len(sources)]
                if source.sample_rate != audio.sample_rate:
                    raise ValueError(f"{noise_type} augmentation sample rate does not match training audio")
                augmented_samples = add_background_noise_snr(audio.samples, source.samples, float(snr_db))
                augmented_features = extract_emotion_features(
                    AudioBuffer(augmented_samples, audio.sample_rate),
                    frame_ms=frame_ms,
                    hop_ms=hop_ms,
                    activity_rms_threshold=activity_rms_threshold,
                )
                features["train"].append(augmented_features)
                train_labels.append(record.emotion.value)
                added += 1
            augmentation_counts[f"{noise_type}@{snr_db:g}dB"] = added

    reverb_augmented = 0
    if augment_reverb:
        rir_sources = noise_sources.get("reverb_impulse", ())
        if not rir_sources:
            raise ValueError("reverb augmentation requested but noise manifest has no reverb_impulse sources")
        for row_index, record in enumerate(accepted_records["train"]):
            audio = load_wav(record.path)
            rir = rir_sources[row_index % len(rir_sources)]
            if rir.sample_rate != audio.sample_rate:
                raise ValueError("reverb impulse response sample rate does not match training audio")
            reverberant = apply_reverb(audio.samples, rir.samples)
            features["train"].append(extract_emotion_features(
                AudioBuffer(reverberant, audio.sample_rate),
                frame_ms=frame_ms,
                hop_ms=hop_ms,
                activity_rms_threshold=activity_rms_threshold,
            ))
            train_labels.append(record.emotion.value)
            reverb_augmented += 1
    train_labels = tuple(train_labels)
    validation_labels = tuple(record.emotion.value for record in accepted_records["validation"])
    missing_train = sorted(set(all_classes) - set(train_labels))
    missing_validation = sorted(set(all_classes) - set(validation_labels))
    if missing_train:
        raise ValueError(f"quality-gated training split is missing classes: {', '.join(missing_train)}")
    if missing_validation:
        raise ValueError(f"quality-gated validation split is missing classes required for calibration: {', '.join(missing_validation)}")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    models = {}
    for profile in ("simple", "full"):
        matrices = {
            name: _feature_matrix(profile, feature_rows)
            for name, feature_rows in features.items()
        }
        artifact = _fit_profile(
            profile,
            matrices["train"],
            train_labels,
            matrices["validation"],
            validation_labels,
            split_metadata=split_metadata,
            seed=seed,
            sample_rate=sample_rate,
            frame_ms=frame_ms,
            hop_ms=hop_ms,
            activity_rms_threshold=activity_rms_threshold,
            training_manifest_sha256=hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
            label_mapping=manifest.label_mapping,
        )
        artifact["training"] = {
            "sample_count": len(train_labels),
            "original_sample_count": len(accepted_records["train"]),
            "augmented_sample_count": len(train_labels) - len(accepted_records["train"]),
            "class_counts": _class_counts(train_labels),
            "split_method": split.method,
            "seed": seed,
            "white_noise_augmentation_snr_db": [float(value) for value in white_noise_augmentation_snr_db],
            "background_noise_augmentation_snr_db": [float(value) for value in background_noise_augmentation_snr_db],
            "background_noise_manifest_sha256": noise_metadata.get("manifest_sha256"),
            "background_noise_source_sha256": noise_metadata.get("source_sha256", {}),
            "reverb_augmentation_enabled": augment_reverb,
            "reverb_augmented_sample_count": reverb_augmented,
            "augmented_rows_by_snr": dict(augmentation_counts),
        }
        artifact["validation"].update({
            "class_counts": _class_counts(validation_labels),
            "rejected_quality_counts": dict(rejects["validation"]),
        })
        filename = f"{profile}.json"
        model_path = output / filename
        model_size = _write_json_atomically(model_path, artifact)
        model_parameter_count = _parameter_count(artifact)
        models[profile] = {
            "path": str(model_path),
            "model_size_bytes": model_size,
            "parameter_count": model_parameter_count,
            "class_counts": artifact["training"]["class_counts"],
            "validation": artifact["validation"],
            "test_metrics_computed": False,
        }

    return {
        "status": "TRAINED_VALIDATION_CALIBRATED",
        "real_performance_evaluated": False,
        "performance_status": "NOT_EVALUATED_TEST_SET",
        "synthetic_data_warning": "Synthetic recordings validate code paths only; they are not emotion recognition evidence.",
        "split": split_metadata,
        "quality_reject_counts": {name: dict(counts) for name, counts in rejects.items()},
        "augmentation": {
            "white_noise_snr_db": [float(value) for value in white_noise_augmentation_snr_db],
            "background_noise_snr_db": [float(value) for value in background_noise_augmentation_snr_db],
            "background_noise_manifest_sha256": noise_metadata.get("manifest_sha256"),
            "background_noise_sources_sha256": noise_metadata.get("source_sha256", {}),
            "reverb_augmented_sample_count": reverb_augmented,
            "augmented_rows_by_snr": dict(augmentation_counts),
        },
        "models": models,
        "training_manifest_sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
    }


def _extract_acceptable_features(
    records,
    *,
    sample_rate: int,
    frame_ms: float,
    hop_ms: float,
    activity_rms_threshold: float,
) -> tuple[list[EmotionFeatures], list, Counter[str]]:
    feature_rows: list[EmotionFeatures] = []
    accepted_records = []
    rejected: Counter[str] = Counter()
    for record in records:
        audio = load_wav(record.path)
        if audio.sample_rate != sample_rate:
            raise ValueError(
                f"sample rate mismatch for {record.path.name}: expected {sample_rate}, got {audio.sample_rate}"
            )
        quality = extract_features(
            audio,
            frame_ms=frame_ms,
            activity_rms_threshold=activity_rms_threshold,
        )
        if quality.quality_status is not AudioQualityStatus.ACCEPTABLE:
            rejected[quality.quality_status.value] += 1
            continue
        features = extract_emotion_features(
            audio,
            frame_ms=frame_ms,
            hop_ms=hop_ms,
            activity_rms_threshold=activity_rms_threshold,
        )
        if features.valid_frame_fraction < 0.95:
            rejected["INVALID_FRAME_CONTENT"] += 1
            continue
        feature_rows.append(features)
        accepted_records.append(record)
    return feature_rows, accepted_records, rejected


def _feature_matrix(profile: str, rows: list[EmotionFeatures]) -> np.ndarray:
    if profile == "simple":
        return np.vstack([row.simple_values for row in rows]).astype(np.float64)
    return np.vstack([row.values for row in rows]).astype(np.float64)


def _fit_profile(
    profile: str,
    train_x: np.ndarray,
    train_labels: tuple[str, ...],
    validation_x: np.ndarray,
    validation_labels: tuple[str, ...],
    *,
    split_metadata: dict,
    seed: int,
    sample_rate: int,
    frame_ms: float,
    hop_ms: float,
    activity_rms_threshold: float,
    training_manifest_sha256: str,
    label_mapping: dict[str, str],
) -> dict:
    try:
        from sklearn import __version__ as sklearn_version
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import LinearSVC
        from sklearn.linear_model import LogisticRegression
    except ImportError as error:
        raise RuntimeError("linear model training requires scikit-learn; NumPy inference does not") from error

    names = SIMPLE_FEATURE_NAMES if profile == "simple" else EMOTION_FEATURE_NAMES
    schema = SIMPLE_FEATURE_SCHEMA_VERSION if profile == "simple" else EMOTION_FEATURE_SCHEMA_VERSION
    classes = tuple(sorted(set(train_labels)))
    imputation = _training_imputation_values(train_x, profile)
    train_filled = _impute_matrix(train_x, imputation)
    validation_filled = _impute_matrix(validation_x, imputation)
    scaler = StandardScaler().fit(train_filled)
    train_scaled = scaler.transform(train_filled)
    validation_scaled = scaler.transform(validation_filled)
    classifier = LinearSVC(random_state=seed, dual="auto").fit(train_scaled, np.asarray(train_labels))
    if tuple(classifier.classes_) != classes:
        raise RuntimeError("linear model class order changed unexpectedly")
    validation_margins = np.asarray(classifier.decision_function(validation_scaled), dtype=np.float64)
    if validation_margins.ndim == 1:
        validation_margins = validation_margins[:, None]
    if validation_margins.shape != (len(validation_labels), len(classes)):
        raise RuntimeError("unexpected one-vs-rest validation margin shape")

    calibration_slopes = []
    calibration_intercepts = []
    for class_index, label in enumerate(classes):
        binary_labels = (np.asarray(validation_labels) == label).astype(np.int64)
        if np.unique(binary_labels).size != 2:
            raise ValueError(f"validation split needs positive and negative examples for {label!r} calibration")
        calibrator = LogisticRegression(C=1e6, solver="lbfgs", random_state=seed)
        calibrator.fit(validation_margins[:, class_index].reshape(-1, 1), binary_labels)
        calibration_slopes.append(float(calibrator.coef_[0, 0]))
        calibration_intercepts.append(float(calibrator.intercept_[0]))

    validation_probabilities = _calibrate_margins(
        validation_margins,
        np.asarray(calibration_slopes),
        np.asarray(calibration_intercepts),
    )
    threshold, threshold_validation = _select_validation_threshold(validation_labels, validation_probabilities, classes)
    brier, ece = _calibration_diagnostics(validation_labels, validation_probabilities, classes)
    weights = np.asarray(classifier.coef_, dtype=np.float64)
    if weights.shape != (len(classes), len(names)):
        raise RuntimeError("unexpected linear coefficient shape")
    return {
        "format_version": MODEL_FORMAT_VERSION,
        "model_version": MODEL_VERSION,
        "feature_profile": profile,
        "feature_schema_version": schema,
        "feature_names": list(names),
        "sample_rate": int(sample_rate),
        "frame_ms": float(frame_ms),
        "hop_ms": float(hop_ms),
        "activity_rms_threshold": float(activity_rms_threshold),
        "quality_gates": {
            "min_activity_ratio": 0.20,
            "min_contiguous_active_frames": 3,
            "clipping_limit": 0.05,
            "minimum_snr_db": 5.0,
            "acceptable_snr_db": 10.0,
        },
        "classes": list(classes),
        "imputation_values": imputation.tolist(),
        "standardization_mean": np.asarray(scaler.mean_, dtype=np.float64).tolist(),
        "standardization_scale": np.asarray(scaler.scale_, dtype=np.float64).tolist(),
        "weights": weights.tolist(),
        "intercepts": np.asarray(classifier.intercept_, dtype=np.float64).tolist(),
        "calibration_slopes": calibration_slopes,
        "calibration_intercepts": calibration_intercepts,
        "rejection_threshold": threshold,
        "calibration_status": "CALIBRATED",
        "calibration_method": "one-vs-rest sigmoid on independent validation split",
        "training_library": {
            "scikit_learn_version": sklearn_version,
            "classifier": "LinearSVC",
            "C": 1.0,
            "dual": "auto",
            "random_state": seed,
            "standardization": "training split only",
            "imputation": "per-feature training median; fail if a column has no observations",
        },
        "training_manifest_sha256": training_manifest_sha256,
        "label_mapping": dict(label_mapping),
        "label_mapping_sha256": hashlib.sha256(
            json.dumps(label_mapping, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "dataset_split": split_metadata,
        "validation": {
            "sample_count": len(validation_labels),
            "manifest_sha256": training_manifest_sha256,
            "threshold_selected_on": "validation",
            "threshold_selection_method": "maximize F1 for validation correctness; ties choose higher threshold",
            "threshold_metrics": threshold_validation,
            "brier_score": brier,
            "ece": ece,
        },
    }


def _training_imputation_values(matrix: np.ndarray, _profile: str) -> np.ndarray:
    result = np.empty(matrix.shape[1], dtype=np.float64)
    for column in range(matrix.shape[1]):
        finite = matrix[np.isfinite(matrix[:, column]), column]
        if finite.size == 0:
            raise ValueError(f"training feature column {column} has no finite observations")
        result[column] = float(np.median(finite))
    return result


def _impute_matrix(matrix: np.ndarray, imputation: np.ndarray) -> np.ndarray:
    if np.isinf(matrix).any():
        raise ValueError("input feature matrix contains infinite values")
    return np.where(np.isfinite(matrix), matrix, imputation.reshape(1, -1))


def _calibrate_margins(margins: np.ndarray, slopes: np.ndarray, intercepts: np.ndarray) -> np.ndarray:
    logits = np.clip(margins * slopes.reshape(1, -1) + intercepts.reshape(1, -1), -60.0, 60.0)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    return probabilities


def _select_validation_threshold(labels: tuple[str, ...], probabilities: np.ndarray, classes: tuple[str, ...]) -> tuple[float, dict]:
    true = np.asarray(labels)
    predicted = np.asarray(classes)[np.argmax(probabilities, axis=1)]
    confidence = np.max(probabilities, axis=1)
    correct = predicted == true
    candidates = np.unique(np.concatenate(([0.0], confidence, [1.0])))
    best = None
    for threshold in candidates:
        accepted = confidence >= threshold
        tp = int(np.sum(accepted & correct))
        fp = int(np.sum(accepted & ~correct))
        fn = int(np.sum(~accepted & correct))
        denominator = 2 * tp + fp + fn
        f1 = (2.0 * tp / denominator) if denominator else 0.0
        candidate = (f1, float(threshold), accepted)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    assert best is not None
    f1, threshold, accepted = best
    accepted_count = int(np.sum(accepted))
    accepted_errors = int(np.sum(accepted & ~correct))
    return threshold, {
        "correctness_f1": float(f1),
        "accepted_count": accepted_count,
        "coverage": float(np.mean(accepted)),
        "rejection_rate": float(1.0 - np.mean(accepted)),
        "accepted_error_count": accepted_errors,
        "accepted_error_rate": float(accepted_errors / accepted_count) if accepted_count else None,
    }


def _calibration_diagnostics(labels: tuple[str, ...], probabilities: np.ndarray, classes: tuple[str, ...]) -> tuple[float, float]:
    class_indices = {label: index for index, label in enumerate(classes)}
    one_hot = np.zeros_like(probabilities)
    for row, label in enumerate(labels):
        one_hot[row, class_indices[label]] = 1.0
    brier = float(np.mean(np.sum(np.square(probabilities - one_hot), axis=1)))
    confidence = np.max(probabilities, axis=1)
    predicted = np.asarray(classes)[np.argmax(probabilities, axis=1)]
    correctness = (predicted == np.asarray(labels)).astype(np.float64)
    ece = 0.0
    bins = np.linspace(0.0, 1.0, 11)
    for index in range(len(bins) - 1):
        in_bin = (confidence >= bins[index]) & (
            confidence < bins[index + 1] if index < len(bins) - 2 else confidence <= bins[index + 1]
        )
        if in_bin.any():
            ece += float(np.mean(in_bin)) * abs(float(np.mean(correctness[in_bin])) - float(np.mean(confidence[in_bin])))
    return brier, float(ece)


def _class_counts(labels: tuple[str, ...]) -> dict[str, int]:
    counts = Counter(labels)
    return {label: int(counts.get(label, 0)) for label in _CLASS_NAMES}


def _parameter_count(artifact: dict) -> int:
    return int(
        np.asarray(artifact["weights"]).size
        + np.asarray(artifact["intercepts"]).size
        + np.asarray(artifact["standardization_mean"]).size
        + np.asarray(artifact["standardization_scale"]).size
        + np.asarray(artifact["imputation_values"]).size
        + np.asarray(artifact["calibration_slopes"]).size
        + np.asarray(artifact["calibration_intercepts"]).size
    )


def _write_json_atomically(path: Path, artifact: dict) -> int:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        serialized = json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        temporary.write_text(serialized, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path.stat().st_size


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="train_linear.py", description="Train validation-calibrated NumPy-exportable linear baselines")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--frame-ms", type=float, default=25.0)
    parser.add_argument("--hop-ms", type=float, default=10.0)
    parser.add_argument("--activity-rms-threshold", type=float, default=0.01)
    parser.add_argument("--augment-white-noise-snr", type=float, nargs="*", default=())
    parser.add_argument("--augmentation-noise-manifest")
    parser.add_argument("--augment-background-snr", type=float, nargs="*", default=())
    parser.add_argument("--augment-reverb", action="store_true")
    parser.add_argument("--label-map")
    args = parser.parse_args(argv)
    try:
        label_map = json.loads(Path(args.label_map).read_text(encoding="utf-8")) if args.label_map else None
        if label_map is not None and not isinstance(label_map, dict):
            raise ValueError("label map JSON must be an object")
        report = train_linear_models(
            args.manifest,
            args.output_dir,
            seed=args.seed,
            sample_rate=args.sample_rate,
            frame_ms=args.frame_ms,
            hop_ms=args.hop_ms,
            activity_rms_threshold=args.activity_rms_threshold,
            white_noise_augmentation_snr_db=args.augment_white_noise_snr,
            background_noise_manifest_path=args.augmentation_noise_manifest,
            background_noise_augmentation_snr_db=args.augment_background_snr,
            augment_reverb=args.augment_reverb,
            label_map=label_map,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"train_linear.py: {error}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0
