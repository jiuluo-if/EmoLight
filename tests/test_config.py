import json

from emolight.config import load_lighting_config
from emolight.events import Emotion, EmotionEvent, EventSource, SystemStatus
from emolight.lighting.policy import EmotionLightingPolicy


def test_lighting_configuration_is_loaded_from_json_and_applied(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"lighting": {"max_brightness": 0.18, "night_max_brightness": 0.08, "transition_ms": 2200}}), encoding="utf-8")
    config = load_lighting_config(path)
    event = EmotionEvent(Emotion.HAPPY, SystemStatus.TARGET_ACTIVE, 0.9, 0.9, 0.9)

    command = EmotionLightingPolicy(config).evaluate(event)
    assert command is not None
    assert command.brightness == 0.18
    assert command.transition_ms == 2200


def test_user_can_customize_emotion_colors(tmp_path):
    path = tmp_path / "colors.json"
    path.write_text(json.dumps({"lighting": {"happy_rgb": [10, 20, 30]}}), encoding="utf-8")
    policy = EmotionLightingPolicy(load_lighting_config(path))
    event = EmotionEvent(
        Emotion.HAPPY,
        SystemStatus.TARGET_ACTIVE,
        0.9,
        0.9,
        0.9,
        source=EventSource.SIMULATION,
    )

    command = policy.evaluate_simulation(event)
    assert command is not None
    assert command.rgb == (10, 20, 30)
