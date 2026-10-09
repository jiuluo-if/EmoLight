import numpy as np

from emolight.audio.ring_buffer import AudioRingBuffer
from emolight.audio.wav import AudioBuffer
from emolight.audio.vad import ActivityFrame, EnergyVAD
from emolight.emotion.predictor import EmotionPredictor, UnconfiguredEmotionPredictor
from emolight.events import EmotionEvent, SystemStatus
from emolight.features.acoustic import AcousticFeatures, extract_features


class RealtimeFeatureRuntime:
    """Bounded rolling feature path; emotion remains unconfigured by default."""

    def __init__(
        self,
        sample_rate: int = 16000,
        window_s: float = 1.5,
        update_interval_s: float = 0.5,
        predictor: EmotionPredictor | None = None,
    ) -> None:
        if sample_rate <= 0 or window_s <= 0 or update_interval_s <= 0:
            raise ValueError("sample_rate, window_s, and update_interval_s must be positive")
        self.sample_rate = sample_rate
        self.window_samples = max(1, round(sample_rate * window_s))
        self.update_interval_ms = max(1, round(update_interval_s * 1000))
        self.buffer = AudioRingBuffer(self.window_samples)
        self.predictor = predictor or UnconfiguredEmotionPredictor()
        self.latest_features: AcousticFeatures | None = None
        self.latest_vad_frames: list[ActivityFrame] = []
        self.vad = EnergyVAD()
        self._last_update_ms: int | None = None

    def feed(self, samples: np.ndarray, timestamp_ms: int) -> EmotionEvent | None:
        self.buffer.append(samples)
        if self.buffer.size < self.window_samples:
            return None
        if self._last_update_ms is not None and timestamp_ms - self._last_update_ms < self.update_interval_ms:
            return None
        self._last_update_ms = timestamp_ms
        audio = AudioBuffer(self.buffer.snapshot(), self.sample_rate)
        self.latest_features = extract_features(audio)
        self.latest_vad_frames = self.vad.process(audio)
        if not any(frame.status is SystemStatus.UNCERTAIN for frame in self.latest_vad_frames):
            return EmotionEvent.unknown(SystemStatus.SILENCE, timestamp_ms=timestamp_ms)
        return self.predictor.predict(self.latest_features, timestamp_ms=timestamp_ms)
