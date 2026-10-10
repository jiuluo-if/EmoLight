import argparse
from dataclasses import asdict
import json
import sys
import time
from typing import Sequence

from emolight.demo import make_demo_event
from emolight.emotion.predictor import EmotionPrediction, ModelStatus, UnconfiguredEmotionPredictor
from emolight.emotion.linear import NumpyLinearEmotionPredictor
from emolight.events import Emotion, EmotionEvent, SystemStatus
from emolight.audio.wav import load_wav
from emolight.config import AppConfig, AudioConfig, load_app_config
from emolight.features.acoustic import AudioQualityStatus, extract_features
from emolight.features.prosody import extract_prosody
from emolight.lighting.policy import EmotionLightingPolicy


def _extract_configured_features(audio, config: AudioConfig):
    return extract_features(
        audio,
        frame_ms=config.frame_ms,
        activity_rms_threshold=config.activity_rms_threshold,
        min_activity_ratio=config.min_activity_ratio,
        min_contiguous_active_frames=config.min_contiguous_active_frames,
        clipping_limit=config.clipping_limit,
        minimum_snr_db=config.minimum_snr_db,
        acceptable_snr_db=config.acceptable_snr_db,
    )


def run(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="emolight", description="Offline EmoLight PC prototype")
    parser.add_argument("--demo", action="store_true", help="emit an explicitly simulated emotion event")
    parser.add_argument("--no-gui", action="store_true", help="print the current event as JSON")
    parser.add_argument("--emotion", choices=[emotion.value for emotion in Emotion], default="happy")
    parser.add_argument("--wav", help="analyze a local PCM WAV file without emotion inference")
    parser.add_argument("--config", help="JSON configuration file for audio, models, and lighting")
    parser.add_argument("--emotion-only", action="store_true", help="offline emotion classification without speaker identity verification")
    parser.add_argument("--model", help="local validation-calibrated NumPy JSON model; requires --emotion-only")
    args = parser.parse_args(argv)
    if args.emotion_only and (args.demo or not args.no_gui or not args.wav or not args.model):
        parser.error("--emotion-only requires --no-gui, --wav, and --model, and cannot be combined with --demo")
    if args.model and not args.emotion_only:
        parser.error("--model requires the explicit --emotion-only mode")
    if args.demo and args.wav:
        parser.error("--demo and --wav are separate input modes")
    try:
        app_config = load_app_config(args.config) if args.config else AppConfig()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"emolight: cannot read config ({type(error).__name__})", file=sys.stderr)
        return 2
    policy = EmotionLightingPolicy(app_config.lighting)
    acoustic_features = None

    if args.emotion_only:
        try:
            audio = load_wav(args.wav)
        except (OSError, ValueError, EOFError) as error:
            print(f"emolight: cannot analyze WAV ({type(error).__name__})", file=sys.stderr)
            return 2
        audio_config = app_config.audio
        if audio.sample_rate != audio_config.sample_rate_hz:
            print(
                f"emolight: WAV sampling rate {audio.sample_rate} does not match configured rate {audio_config.sample_rate_hz}",
                file=sys.stderr,
            )
            return 2
        predictor = NumpyLinearEmotionPredictor.load(
            args.model,
            expected_sample_rate=audio_config.sample_rate_hz,
            expected_frame_ms=audio_config.frame_ms,
            expected_hop_ms=audio_config.hop_ms,
            expected_activity_rms_threshold=audio_config.activity_rms_threshold,
        )
        acoustic_features = _extract_configured_features(audio, audio_config)
        prosody = extract_prosody(
            audio,
            frame_ms=audio_config.frame_ms,
            hop_ms=audio_config.hop_ms,
            max_f0_hz=500.0 if audio.sample_rate > 1000 else audio.sample_rate * 0.45,
            activity_rms_threshold=audio_config.activity_rms_threshold,
        )
        if acoustic_features.quality_status is AudioQualityStatus.ACCEPTABLE:
            prediction = predictor.predict(prosody, timestamp_ms=int(time.time() * 1000))
        else:
            rejection = (
                SystemStatus.LOW_QUALITY
                if acoustic_features.quality_status is AudioQualityStatus.LOW_QUALITY
                else SystemStatus.UNCERTAIN
            )
            prediction = EmotionPrediction(
                emotion=None,
                model_status=predictor.model_status,
                rejection_status=rejection,
                timestamp_ms=int(time.time() * 1000),
                audio_quality=acoustic_features.audio_quality,
                model_version=predictor.metadata.get("model_version"),
                calibration_status=predictor.metadata.get("calibration_status", "UNCALIBRATED"),
                decision_threshold=predictor.metadata.get("rejection_threshold"),
                validation_record_id=predictor.validation_record_id,
            )
        payload = {
            "mode": "OFFLINE_EMOTION_ONLY",
            "identity_status": "NOT_EVALUATED",
            "target_event": None,
            "prediction": {
                "status": prediction.rejection_status.value if prediction.rejection_status else prediction.model_status.value,
                "model_status": prediction.model_status.value,
                "rejection_status": prediction.rejection_status.value if prediction.rejection_status else None,
                "model_version": prediction.model_version,
                "calibration_status": prediction.calibration_status,
                "decision_threshold": prediction.decision_threshold,
                "validation_record_id": prediction.validation_record_id,
                "quality_status": acoustic_features.quality_status.value,
                "emotion": prediction.emotion.value if prediction.emotion else None,
                "confidence": prediction.confidence,
                "scores": {emotion.value: score for emotion, score in prediction.scores.items()},
                "score_semantics": "Calibrated model score only; not verified target identity or a clinical emotion probability.",
            },
            "acoustic_features": asdict(acoustic_features),
            "lighting": None,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.demo:
        event = make_demo_event(Emotion(args.emotion), timestamp_ms=int(time.time() * 1000))
        command = policy.evaluate_simulation(event)
        mode = "SIMULATION"
    else:
        if args.wav:
            try:
                audio = load_wav(args.wav)
                acoustic_features = _extract_configured_features(audio, app_config.audio)
            except (OSError, ValueError, EOFError) as error:
                print(f"emolight: cannot read WAV ({type(error).__name__})", file=sys.stderr)
                return 2
            event = UnconfiguredEmotionPredictor().predict(acoustic_features, int(time.time() * 1000))
        else:
            event = EmotionEvent.unknown(SystemStatus.NOT_CONFIGURED, int(time.time() * 1000))
        command = policy.evaluate(event)
        mode = "LIVE_UNCONFIGURED"

    if args.no_gui:
        payload = {
            "mode": mode,
            "event": {
                "emotion": event.emotion.value if event.emotion else None,
                "status": event.status.value,
                "emotion_confidence": event.emotion_confidence,
                "speaker_confidence": event.speaker_confidence,
                "audio_quality": event.audio_quality,
                "timestamp_ms": event.timestamp_ms,
                "source": event.source.value,
            },
            "emotion_scores": ({emotion.value: (0.95 if emotion is event.emotion else 0.0) for emotion in Emotion} if args.demo else None),
            "acoustic_features": asdict(acoustic_features) if acoustic_features is not None else None,
            "lighting": asdict(command) if command is not None else None,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    from emolight.gui.app import launch

    launch(app_config)
    return 0


def main() -> int:
    return run()
