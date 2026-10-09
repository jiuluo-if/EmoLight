import argparse
from dataclasses import asdict
import json
import sys
import time
from typing import Sequence

from emolight.demo import make_demo_event
from emolight.emotion.predictor import UnconfiguredEmotionPredictor
from emolight.events import Emotion, EmotionEvent, SystemStatus
from emolight.audio.wav import load_wav
from emolight.config import LightingConfig, load_lighting_config
from emolight.features.acoustic import extract_features
from emolight.lighting.policy import EmotionLightingPolicy


def run(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="emolight", description="Offline EmoLight PC prototype")
    parser.add_argument("--demo", action="store_true", help="emit an explicitly simulated emotion event")
    parser.add_argument("--no-gui", action="store_true", help="print the current event as JSON")
    parser.add_argument("--emotion", choices=[emotion.value for emotion in Emotion], default="happy")
    parser.add_argument("--wav", help="analyze a local PCM WAV file without emotion inference")
    parser.add_argument("--config", help="JSON configuration file; reads its lighting section")
    args = parser.parse_args(argv)
    if args.demo and args.wav:
        parser.error("--demo and --wav are separate input modes")
    try:
        lighting_config = load_lighting_config(args.config) if args.config else LightingConfig()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"emolight: cannot read config: {error}", file=sys.stderr)
        return 2
    policy = EmotionLightingPolicy(lighting_config)
    acoustic_features = None

    if args.demo:
        event = make_demo_event(Emotion(args.emotion), timestamp_ms=int(time.time() * 1000))
        command = policy.evaluate_simulation(event)
        mode = "SIMULATION"
    else:
        if args.wav:
            try:
                acoustic_features = extract_features(load_wav(args.wav))
            except (OSError, ValueError, EOFError) as error:
                print(f"emolight: cannot read WAV: {error}", file=sys.stderr)
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

    launch(lighting_config)
    return 0


def main() -> int:
    return run()
