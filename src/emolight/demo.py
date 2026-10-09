from emolight.events import Emotion, EmotionEvent, EventSource, SystemStatus


def make_demo_event(emotion: Emotion, timestamp_ms: int = 0) -> EmotionEvent:
    return EmotionEvent(
        emotion=emotion,
        status=SystemStatus.TARGET_ACTIVE,
        emotion_confidence=0.95,
        speaker_confidence=0.95,
        audio_quality=0.95,
        timestamp_ms=timestamp_ms,
        source=EventSource.SIMULATION,
    )
