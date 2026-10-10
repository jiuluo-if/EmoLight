"""Offline emotion-only WAV diagnostics using the pure NumPy JSON model."""

import argparse
from dataclasses import asdict
import json
import time
from typing import Sequence

from emolight.audio.wav import load_wav
from emolight.config import AppConfig, load_app_config
from emolight.emotion.linear import NumpyLinearEmotionPredictor
from emolight.emotion.predictor import EmotionPrediction
from emolight.events import SystemStatus
from emolight.features.acoustic import AudioQualityStatus, extract_features
from emolight.features.prosody import extract_prosody


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="predict_linear.py", description="Run a local NumPy linear emotion model without speaker verification")
    parser.add_argument("--wav", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config")
    args = parser.parse_args(argv)
    try:
        config = load_app_config(args.config) if args.config else AppConfig()
        audio = load_wav(args.wav)
        audio_config = config.audio
        if audio.sample_rate != audio_config.sample_rate_hz:
            raise ValueError(
                f"WAV sampling rate {audio.sample_rate} does not match configured rate {audio_config.sample_rate_hz}"
            )
        predictor = NumpyLinearEmotionPredictor.load(
            args.model,
            expected_sample_rate=audio_config.sample_rate_hz,
            expected_frame_ms=audio_config.frame_ms,
            expected_hop_ms=audio_config.hop_ms,
            expected_activity_rms_threshold=audio_config.activity_rms_threshold,
        )
        acoustic = extract_features(
            audio,
            frame_ms=audio_config.frame_ms,
            activity_rms_threshold=audio_config.activity_rms_threshold,
            min_activity_ratio=audio_config.min_activity_ratio,
            min_contiguous_active_frames=audio_config.min_contiguous_active_frames,
            clipping_limit=audio_config.clipping_limit,
            minimum_snr_db=audio_config.minimum_snr_db,
            acceptable_snr_db=audio_config.acceptable_snr_db,
        )
        features = extract_prosody(
            audio,
            frame_ms=audio_config.frame_ms,
            hop_ms=audio_config.hop_ms,
            activity_rms_threshold=audio_config.activity_rms_threshold,
        )
    except (OSError, ValueError, EOFError) as error:
        parser.exit(2, f"predict_linear.py: operation failed ({type(error).__name__})\n")

    if acoustic.quality_status is AudioQualityStatus.ACCEPTABLE:
        prediction = predictor.predict(features, timestamp_ms=int(time.time() * 1000))
    else:
        rejection = (
            SystemStatus.LOW_QUALITY
            if acoustic.quality_status is AudioQualityStatus.LOW_QUALITY
            else SystemStatus.UNCERTAIN
        )
        metadata = predictor.metadata
        prediction = EmotionPrediction(
            emotion=None,
            model_status=predictor.model_status,
            rejection_status=rejection,
            timestamp_ms=int(time.time() * 1000),
            audio_quality=acoustic.audio_quality,
            model_version=metadata.get("model_version"),
            calibration_status=metadata.get("calibration_status", "UNCALIBRATED"),
            decision_threshold=metadata.get("rejection_threshold"),
            validation_record_id=predictor.validation_record_id,
        )
    payload = {
        "mode": "OFFLINE_EMOTION_ONLY",
        "identity_status": "NOT_EVALUATED",
        "target_event": None,
        "prediction": {
            "status": prediction.rejection_status.value if prediction.rejection_status else prediction.model_status.value,
            "model_status": prediction.model_status.value,
            "calibration_status": prediction.calibration_status,
            "model_version": prediction.model_version,
            "validation_record_id": prediction.validation_record_id,
            "decision_threshold": prediction.decision_threshold,
            "quality_status": acoustic.quality_status.value,
            "emotion": prediction.emotion.value if prediction.emotion else None,
            "confidence": prediction.confidence,
            "scores": {emotion.value: score for emotion, score in prediction.scores.items()},
            "score_semantics": "Validation-calibrated class score; not target identity or a clinical probability.",
        },
        "acoustic_features": asdict(acoustic),
        "lighting": None,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
