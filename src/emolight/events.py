from dataclasses import dataclass
from enum import Enum


class Emotion(str, Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    ANGRY = "angry"
    SAD = "sad"


class SystemStatus(str, Enum):
    TARGET_ACTIVE = "TARGET_ACTIVE"
    NON_TARGET = "NON_TARGET"
    SILENCE = "SILENCE"
    OVERLAP = "OVERLAP"
    LOW_QUALITY = "LOW_QUALITY"
    UNCERTAIN = "UNCERTAIN"
    NOT_CONFIGURED = "NOT_CONFIGURED"


class EventSource(str, Enum):
    LIVE = "LIVE"
    SIMULATION = "SIMULATION"


@dataclass(frozen=True)
class EmotionEvent:
    emotion: Emotion | None
    status: SystemStatus
    emotion_confidence: float = 0.0
    speaker_confidence: float = 0.0
    audio_quality: float = 0.0
    timestamp_ms: int = 0
    source: EventSource = EventSource.LIVE
    model_version: str | None = None
    calibration_status: str | None = None
    decision_threshold: float | None = None
    validation_record_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("emotion_confidence", "speaker_confidence", "audio_quality"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be non-negative")
        if self.status is SystemStatus.NOT_CONFIGURED and self.emotion is not None:
            raise ValueError("NOT_CONFIGURED events cannot contain an emotion")
        if self.decision_threshold is not None and not 0.0 <= self.decision_threshold <= 1.0:
            raise ValueError("decision_threshold must be between 0 and 1")

    @classmethod
    def unknown(cls, status: SystemStatus, timestamp_ms: int = 0) -> "EmotionEvent":
        return cls(emotion=None, status=status, timestamp_ms=timestamp_ms)
