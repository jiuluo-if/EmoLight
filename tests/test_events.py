import pytest

from emolight.config import LightingConfig
from emolight.events import Emotion, EmotionEvent, EventSource, SystemStatus


def test_missing_model_uses_not_configured_without_emotion():
    event = EmotionEvent.unknown(SystemStatus.NOT_CONFIGURED, timestamp_ms=12)

    assert event.emotion is None
    assert event.source is EventSource.LIVE
    assert event.timestamp_ms == 12


def test_simulation_is_explicit_and_confidences_are_bounded():
    event = EmotionEvent(
        emotion=Emotion.HAPPY,
        status=SystemStatus.TARGET_ACTIVE,
        emotion_confidence=0.9,
        speaker_confidence=1.0,
        audio_quality=0.8,
        timestamp_ms=100,
        source=EventSource.SIMULATION,
    )

    assert event.source.value == "SIMULATION"
    with pytest.raises(ValueError):
        EmotionEvent(
            emotion=Emotion.HAPPY,
            status=SystemStatus.TARGET_ACTIVE,
            emotion_confidence=1.1,
            speaker_confidence=1.0,
            audio_quality=0.8,
            timestamp_ms=100,
        )


def test_lighting_defaults_have_safe_limits():
    config = LightingConfig()

    assert config.max_brightness <= 0.4
    assert config.night_max_brightness < config.max_brightness
    assert config.transition_ms >= 1500
    assert config.allow_flicker is False
