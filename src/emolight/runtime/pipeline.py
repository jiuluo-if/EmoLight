from dataclasses import dataclass, replace

import numpy as np

from emolight.audio.ring_buffer import AudioRingBuffer
from emolight.audio.wav import AudioBuffer
from emolight.emotion.predictor import EmotionPrediction, EmotionPredictor, ModelStatus, UnconfiguredEmotionPredictor
from emolight.events import EmotionEvent, EventSource, SystemStatus
from emolight.features.acoustic import AcousticFeatures, AudioQualityStatus, extract_features
from emolight.features.prosody import ProsodyFeatures, extract_prosody
from emolight.speaker.verifier import SpeakerVerifier, UnconfiguredSpeakerVerifier


@dataclass(frozen=True)
class RuntimeSnapshot:
    event: EmotionEvent
    acoustic_features: AcousticFeatures
    timestamp_ms: int
    prosody_features: ProsodyFeatures | None = None


class RealtimeFeatureRuntime:
    """Bounded rolling feature path; emotion remains unconfigured by default."""

    def __init__(
        self,
        sample_rate: int = 16000,
        window_s: float = 1.5,
        update_interval_s: float = 0.5,
        predictor: EmotionPredictor | None = None,
        speaker_verifier: SpeakerVerifier | None = None,
        frame_ms: float = 25.0,
        hop_ms: float = 10.0,
        activity_rms_threshold: float = 0.01,
        clipping_limit: float = 0.05,
        minimum_snr_db: float = 5.0,
        acceptable_snr_db: float = 10.0,
        min_activity_ratio: float = 0.20,
        min_contiguous_active_frames: int = 3,
        min_audio_quality: float = 0.50,
        min_speaker_confidence: float = 0.80,
    ) -> None:
        if sample_rate <= 0 or window_s <= 0 or update_interval_s <= 0:
            raise ValueError("sample_rate, window_s, and update_interval_s must be positive")
        if frame_ms <= 0 or hop_ms <= 0 or hop_ms > frame_ms or activity_rms_threshold < 0:
            raise ValueError("analysis frame/hop and activity threshold are invalid")
        if not 0.0 <= clipping_limit <= 1.0 or minimum_snr_db >= acceptable_snr_db:
            raise ValueError("clipping and SNR quality thresholds are invalid")
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
        self.frame_ms = frame_ms
        self.hop_ms = hop_ms
        self.activity_rms_threshold = activity_rms_threshold
        self.clipping_limit = clipping_limit
        self.minimum_snr_db = minimum_snr_db
        self.acceptable_snr_db = acceptable_snr_db
        self.min_activity_ratio = min_activity_ratio
        self.min_contiguous_active_frames = min_contiguous_active_frames
        self.min_audio_quality = min_audio_quality
        self.min_speaker_confidence = min_speaker_confidence
        self.latest_features: AcousticFeatures | None = None
        self.latest_prosody: ProsodyFeatures | None = None
        self._last_update_ms: int | None = None

    def feed(self, samples: np.ndarray, timestamp_ms: int) -> EmotionEvent | None:
        snapshot = self.feed_snapshot(samples, timestamp_ms)
        return snapshot.event if snapshot is not None else None

    def feed_snapshot(self, samples: np.ndarray, timestamp_ms: int) -> RuntimeSnapshot | None:
        self.buffer.append(samples)
        if self.buffer.size < self.window_samples:
            return None
        if self._last_update_ms is not None and timestamp_ms - self._last_update_ms < self.update_interval_ms:
            return None
        self._last_update_ms = timestamp_ms
        audio = AudioBuffer(self.buffer.snapshot(), self.sample_rate)
        self.latest_prosody = None
        self.latest_features = extract_features(
            audio,
            frame_ms=self.frame_ms,
            activity_rms_threshold=self.activity_rms_threshold,
            min_activity_ratio=self.min_activity_ratio,
            min_contiguous_active_frames=self.min_contiguous_active_frames,
            clipping_limit=self.clipping_limit,
            minimum_snr_db=self.minimum_snr_db,
            acceptable_snr_db=self.acceptable_snr_db,
        )

        def finish(event: EmotionEvent) -> RuntimeSnapshot:
            if event.audio_quality != self.latest_features.audio_quality:
                event = replace(event, audio_quality=self.latest_features.audio_quality)
            return RuntimeSnapshot(event, self.latest_features, timestamp_ms, self.latest_prosody)

        if self.latest_features.active_frame_count == 0:
            return finish(EmotionEvent.unknown(SystemStatus.SILENCE, timestamp_ms=timestamp_ms))
        if (
            self.latest_features.activity_ratio < self.min_activity_ratio
            or self.latest_features.longest_active_run < self.min_contiguous_active_frames
        ):
            return finish(EmotionEvent.unknown(SystemStatus.UNCERTAIN, timestamp_ms=timestamp_ms))
        if self.latest_features.quality_status is AudioQualityStatus.LOW_QUALITY:
            return finish(EmotionEvent.unknown(SystemStatus.LOW_QUALITY, timestamp_ms=timestamp_ms))
        if self.latest_features.quality_status is not AudioQualityStatus.ACCEPTABLE:
            return finish(EmotionEvent.unknown(SystemStatus.UNCERTAIN, timestamp_ms=timestamp_ms))
        if self.latest_features.audio_quality < self.min_audio_quality:
            return finish(EmotionEvent.unknown(SystemStatus.LOW_QUALITY, timestamp_ms=timestamp_ms))

        try:
            identity = self.speaker_verifier.verify(audio)
        except Exception:
            return finish(EmotionEvent.unknown(SystemStatus.UNCERTAIN, timestamp_ms=timestamp_ms))
        if identity.status is not SystemStatus.TARGET_ACTIVE:
            return finish(EmotionEvent.unknown(identity.status, timestamp_ms=timestamp_ms))
        if identity.confidence < self.min_speaker_confidence:
            return finish(EmotionEvent.unknown(SystemStatus.UNCERTAIN, timestamp_ms=timestamp_ms))

        try:
            self.latest_prosody = extract_prosody(
                audio,
                frame_ms=self.frame_ms,
                hop_ms=self.hop_ms,
                max_f0_hz=500.0 if audio.sample_rate > 1000 else audio.sample_rate * 0.45,
                activity_rms_threshold=self.activity_rms_threshold,
            )
            prediction = self.predictor.predict(self.latest_prosody, timestamp_ms=timestamp_ms)
        except Exception:
            return finish(EmotionEvent.unknown(SystemStatus.UNCERTAIN, timestamp_ms=timestamp_ms))
        prediction = self._normalize_prediction(prediction, timestamp_ms)
        if prediction.model_status is not ModelStatus.READY or prediction.emotion is None:
            return finish(EmotionEvent.unknown(prediction.status, timestamp_ms=timestamp_ms))
        return finish(EmotionEvent(
            emotion=prediction.emotion,
            status=SystemStatus.TARGET_ACTIVE,
            emotion_confidence=prediction.confidence,
            speaker_confidence=identity.confidence,
            audio_quality=self.latest_features.audio_quality,
            timestamp_ms=timestamp_ms,
            source=EventSource.LIVE,
        ))

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
