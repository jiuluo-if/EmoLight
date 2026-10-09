import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from emolight.audio.wav import load_wav
from emolight.data.manifest import load_manifest
from emolight.data.split import DatasetSplit, split_by_speaker_and_recording
from emolight.emotion.sklearn_predictor import fit_calibrated_svm, save_model_artifact
from emolight.events import Emotion
from emolight.features.acoustic import AudioQualityStatus, extract_features
from emolight.features.prosody import extract_prosody


def train_from_manifest(
    manifest_path: str | Path,
    model_path: str | Path,
    *,
    seed: int = 42,
    sample_rate: int = 16000,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    activity_rms_threshold: float = 0.01,
    label_map: Mapping[str, str | Emotion] | None = None,
) -> dict:
    """Train on a local manifest. This function does not report model performance."""
    manifest = load_manifest(manifest_path, label_map=label_map)
    if not manifest.records:
        raise ValueError("manifest has no supported labeled audio rows")
    split: DatasetSplit = split_by_speaker_and_recording(manifest.records, seed=seed)

    vectors: list[np.ndarray] = []
    labels: list[str] = []
    groups: list[int] = []
    skipped_low_quality = 0
    skipped_uncertain_quality = 0
    for record, group_id in zip(split.train, split.train_group_ids):
        audio = load_wav(record.path)
        if audio.sample_rate != sample_rate:
            raise ValueError(
                f"sample rate mismatch for recording {record.recording_id!r}: "
                f"expected {sample_rate}, got {audio.sample_rate}"
            )
        quality = extract_features(audio, frame_ms=frame_ms, activity_rms_threshold=activity_rms_threshold)
        if quality.quality_status is not AudioQualityStatus.ACCEPTABLE:
            if quality.quality_status is AudioQualityStatus.LOW_QUALITY:
                skipped_low_quality += 1
            else:
                skipped_uncertain_quality += 1
            continue
        prosody = extract_prosody(
            audio,
            frame_ms=frame_ms,
            hop_ms=hop_ms,
            activity_rms_threshold=activity_rms_threshold,
        )
        if prosody.voiced_fraction == 0.0:
            skipped_low_quality += 1
            continue
        vectors.append(prosody.values)
        labels.append(record.emotion.value)
        groups.append(group_id)

    if not vectors:
        raise ValueError("all training rows were rejected by audio quality/VAD checks")
    fit_result = fit_calibrated_svm(
        np.vstack(vectors),
        labels,
        groups,
        seed=seed,
        sample_rate=sample_rate,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        activity_rms_threshold=activity_rms_threshold,
        split_metadata={
            **split.to_metadata(),
            "manifest_sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
            "dataset_ids": sorted({record.dataset_id for record in manifest.records}),
            "label_mapping": manifest.label_mapping,
            "label_mapping_sha256": hashlib.sha256(
                json.dumps(manifest.label_mapping, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "unmapped_label_counts": manifest.skipped_labels,
        },
    )
    model_size = save_model_artifact(model_path, fit_result)
    return {
        "status": "TRAINED",
        "model_path": str(Path(model_path)),
        "model_size_bytes": model_size,
        "train_rows_used": len(vectors),
        "train_rows_rejected": skipped_low_quality,
        "train_rows_rejected_uncertain_quality": skipped_uncertain_quality,
        "class_counts": dict(Counter(labels)),
        "split": split.to_metadata(),
        "real_performance_evaluated": False,
        "performance_status": "NOT_EVALUATED_NO_HELD_OUT_EVALUATION",
        "synthetic_data_warning": "Synthetic recordings may test this script but are not evidence of emotion recognition performance.",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="emolight-train", description="Train the local calibrated prosody SVM baseline")
    parser.add_argument("--manifest", required=True, help="CSV manifest with path, emotion, speaker_id, recording_id columns")
    parser.add_argument("--model", required=True, help="output path for a trusted local joblib model artifact")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--frame-ms", type=float, default=25.0)
    parser.add_argument("--hop-ms", type=float, default=10.0)
    parser.add_argument("--activity-rms-threshold", type=float, default=0.01)
    parser.add_argument("--label-map", help="JSON object mapping corpus labels to neutral/happy/angry/sad")
    args = parser.parse_args(argv)
    try:
        label_map = json.loads(Path(args.label_map).read_text(encoding="utf-8")) if args.label_map else None
        if label_map is not None and not isinstance(label_map, dict):
            raise ValueError("label map JSON must be an object")
        result = train_from_manifest(
            args.manifest,
            args.model,
            seed=args.seed,
            sample_rate=args.sample_rate,
            frame_ms=args.frame_ms,
            hop_ms=args.hop_ms,
            activity_rms_threshold=args.activity_rms_threshold,
            label_map=label_map,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"emolight-train: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
