from typing import Protocol

from emolight.events import EmotionEvent, SystemStatus
from emolight.features.acoustic import AcousticFeatures


class EmotionPredictor(Protocol):
    def predict(self, features: AcousticFeatures, timestamp_ms: int = 0) -> EmotionEvent: ...


class UnconfiguredEmotionPredictor:
    """Fail-closed predictor used until a trained and verified model is installed."""

    def predict(self, features: AcousticFeatures, timestamp_ms: int = 0) -> EmotionEvent:
        status = SystemStatus.LOW_QUALITY if features.audio_quality < 0.5 else SystemStatus.NOT_CONFIGURED
        return EmotionEvent.unknown(status, timestamp_ms=timestamp_ms)
