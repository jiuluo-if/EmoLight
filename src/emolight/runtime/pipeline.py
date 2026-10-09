import numpy as np

from emolight.audio.ring_buffer import AudioRingBuffer
from emolight.audio.wav import AudioBuffer
from emolight.audio.vad import ActivityFrame, EnergyVAD
from emolight.emotion.predictor import EmotionPrediction, EmotionPredictor, ModelStatus, UnconfiguredEmotionPredictor
from emolight.events import EmotionEvent, EventSource, SystemStatus
from emolight.features.acoustic import AcousticFeatures, extract_features
from emolight.speaker.verifier import SpeakerVerifier, UnconfiguredSpeakerVerifier


class RealtimeFeatureRuntime:
    """Bounded rolling feature path; emotion remains unconfigured by default."""

    def __init__(
        self,
        sample_rate: int = 16000,
        window_s: float = 1.5,
        update_interval_s: float = 0.5,
        predictor: EmotionPredictor | None = None,
        speaker_verifier: SpeakerVerifier | None = None,
        min_activity_ratio: float = 0.20,
        min_contiguous_active_frames: int = 3,
        min_audio_quality: float = 0.50,
        min_speaker_confidence: float = 0.80,
    ) -> None:
        if sample_rate <= 0 or window_s <= 0 or update_interval_s <= 0:
            raise ValueError("sample_rate, window_s, and update_interval_s must be positive")
        if not 0.0 <= min_activity_ratio <= 1.0 or min_contiguous_active_frames < 1:
            raise ValueError("activity ratio must be bounded and contiguous frame count positive")
        if not 0.0 <= min_audio_quality <= 1.0 or not 0.0 <= min_speaker_confidence <= 1.0:
            raise ValueError("quality and speaker confidence thresholds must be between 0 and 1")
        self.sample_rate = sample_rate
        self.window_samples = max(1, round(sample_rate * window_s))
        self.update_interval_ms = max(1, round(update_interval_s * 1000))
        self.buffer = AudioRingBuffer(self.window_samples)
        self.predictor = predictor or UnconfiguredEmotionPredictor()
        self.speaker_verifier = speaker_verifier or UnconfiguredSpeakerVerifier()
        self.min_activity_ratio = min_activity_ratio
        self.min_contiguous_active_frames = min_contiguous_active_frames
        self.min_audio_quality = min_audio_quality
        self.min_speaker_confidence = min_speaker_confidence
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
        active = [frame.status is SystemStatus.UNCERTAIN for frame in self.latest_vad_frames]
        activity_ratio = sum(active) / len(active) if active else 0.0
        longest_active_run = max((count for value, count in _group_runs(active) if value), default=0)
        if activity_ratio == 0.0:
            return EmotionEvent.unknown(SystemStatus.SILENCE, timestamp_ms=timestamp_ms)
        if activity_ratio < self.min_activity_ratio or longest_active_run < self.min_contiguous_active_frames:
            return EmotionEvent.unknown(SystemStatus.UNCERTAIN, timestamp_ms=timestamp_ms)
        if self.latest_features.audio_quality < self.min_audio_quality:
            return EmotionEvent.unknown(SystemStatus.LOW_QUALITY, timestamp_ms=timestamp_ms)

        try:
            identity = self.speaker_verifier.verify(audio)
        except Exception:
            return EmotionEvent.unknown(SystemStatus.UNCERTAIN, timestamp_ms=timestamp_ms)
        if identity.status is not SystemStatus.TARGET_ACTIVE:
            return EmotionEvent.unknown(identity.status, timestamp_ms=timestamp_ms)
        if identity.confidence < self.min_speaker_confidence:
            return EmotionEvent.unknown(SystemStatus.UNCERTAIN, timestamp_ms=timestamp_ms)

        try:
            prediction = self.predictor.predict(self.latest_features, timestamp_ms=timestamp_ms)
        except Exception:
            return EmotionEvent.unknown(SystemStatus.UNCERTAIN, timestamp_ms=timestamp_ms)
        prediction = self._normalize_prediction(prediction, timestamp_ms)
        if prediction.model_status is not ModelStatus.READY or prediction.emotion is None:
            return EmotionEvent.unknown(prediction.status, timestamp_ms=timestamp_ms)
        return EmotionEvent(
            emotion=prediction.emotion,
            status=SystemStatus.TARGET_ACTIVE,
            emotion_confidence=prediction.confidence,
            speaker_confidence=identity.confidence,
            audio_quality=self.latest_features.audio_quality,
            timestamp_ms=timestamp_ms,
            source=EventSource.LIVE,
        )

    @staticmethod
    def _normalize_prediction(value, timestamp_ms: int) -> EmotionPrediction:
        if isinstance(value, EmotionPrediction):
            return value
        if isinstance(value, EmotionEvent):
            if value.emotion is None:
                return EmotionPrediction(
                    emotion=None,
                    model_status=ModelStatus.NOT_CONFIGURED if value.status is SystemStatus.NOT_CONFIGURED else ModelStatus.READY,
                    rejection_status=value.status if value.status in (SystemStatus.LOW_QUALITY, SystemStatus.UNCERTAIN) else None,
                    timestamp_ms=timestamp_ms,
                )
            # Compatibility with older classifiers: ignore their identity/source fields.
            return EmotionPrediction(
                emotion=value.emotion,
                confidence=value.emotion_confidence,
                model_status=ModelStatus.READY,
                timestamp_ms=timestamp_ms,
            )
        return EmotionPrediction(
            emotion=None,
            model_status=ModelStatus.ERROR,
            rejection_status=SystemStatus.UNCERTAIN,
            timestamp_ms=timestamp_ms,
        )


def _group_runs(values: list[bool]):
    if not values:
        return
    current = values[0]
    count = 0
    for value in values:
        if value == current:
            count += 1
        else:
            yield current, count
            current = value
            count = 1
    yield current, count
