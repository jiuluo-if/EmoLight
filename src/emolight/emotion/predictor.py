from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Protocol

from emolight.events import Emotion, EmotionEvent, EventSource, SystemStatus
from emolight.features.acoustic import AcousticFeatures, AudioQualityStatus


class ModelStatus(str, Enum):
    READY = "READY"
    NOT_CONFIGURED = "NOT_CONFIGURED"
    INVALID = "INVALID"
    ERROR = "ERROR"


@dataclass(frozen=True)
class EmotionPrediction:
    """Emotion-only classifier output; it deliberately carries no identity verdict."""

    emotion: Emotion | None
    confidence: float = 0.0
    scores: Mapping[Emotion, float] = field(default_factory=dict)
    model_status: ModelStatus = ModelStatus.READY
    rejection_status: SystemStatus | None = None
    timestamp_ms: int = 0
    audio_quality: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if any(not 0.0 <= value <= 1.0 for value in self.scores.values()):
            raise ValueError("emotion scores must be between 0 and 1")
        if self.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be non-negative")
        if not 0.0 <= self.audio_quality <= 1.0:
            raise ValueError("audio_quality must be between 0 and 1")
        if self.rejection_status is not None and self.rejection_status not in (
            SystemStatus.LOW_QUALITY,
            SystemStatus.UNCERTAIN,
            SystemStatus.NOT_CONFIGURED,
        ):
            raise ValueError("emotion predictions cannot carry speaker or VAD status")
        if self.model_status is not ModelStatus.READY and self.emotion is not None:
            raise ValueError("non-ready model predictions cannot contain an emotion")

    @property
    def emotion_confidence(self) -> float:
        return self.confidence

    @property
    def status(self) -> SystemStatus:
        if self.rejection_status is not None:
            return self.rejection_status
        if self.model_status in (ModelStatus.NOT_CONFIGURED, ModelStatus.INVALID):
            return SystemStatus.NOT_CONFIGURED
        return SystemStatus.UNCERTAIN

    @property
    def speaker_confidence(self) -> float:
        return 0.0

    @property
    def source(self) -> EventSource:
        return EventSource.LIVE


class EmotionPredictor(Protocol):
    def predict(self, features: AcousticFeatures, timestamp_ms: int = 0) -> EmotionPrediction: ...


class UnconfiguredEmotionPredictor:
    """Fail-closed predictor used until a trained and verified model is installed."""

    def predict(self, features: AcousticFeatures, timestamp_ms: int = 0) -> EmotionPrediction:
        rejection = SystemStatus.LOW_QUALITY if features.quality_status is AudioQualityStatus.LOW_QUALITY else None
        return EmotionPrediction(
            emotion=None,
            model_status=ModelStatus.NOT_CONFIGURED,
            rejection_status=rejection,
            timestamp_ms=timestamp_ms,
            audio_quality=features.audio_quality,
        )
