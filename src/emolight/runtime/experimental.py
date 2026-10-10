"""Realtime emotion-only inference, explicitly separated from target events."""

from collections import deque
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from emolight.audio.ring_buffer import AudioRingBuffer
from emolight.audio.wav import AudioBuffer
from emolight.emotion.predictor import EmotionPredictor, ModelStatus, UnconfiguredEmotionPredictor
from emolight.events import Emotion, SystemStatus
from emolight.features.acoustic import AudioQualityStatus, AcousticFeatures, extract_features
from emolight.features.emotion_v1 import EmotionFeatures, extract_emotion_features


EXPERIMENTAL_MODE = "EMOTION_ONLY_EXPERIMENTAL"


@dataclass(frozen=True)
class EmotionOnlyResult:
    mode: str
    identity_status: str
    emotion: Emotion | None
    candidate_emotion: Emotion | None
    confidence: float
    scores: Mapping[Emotion, float]
    rejection_reason: str | None
    audio_active: bool
    activity_ratio: float
    audio_quality: float
    audio_quality_status: str
    timestamp_ms: int
    model_version: str | None
    calibration_status: str
    rejection_threshold: float | None
    audio_features: AcousticFeatures
    emotion_features: EmotionFeatures | None


class EmotionOnlyExperimentalRuntime:
    """Classify current audio without speaker identity or target-event semantics."""

    def __init__(
        self,
        predictor: EmotionPredictor | None = None,
        *,
        sample_rate: int = 16000,
        window_s: float | None = None,
        update_interval_s: float | None = None,
        frame_ms: float | None = None,
        hop_ms: float | None = None,
        activity_rms_threshold: float | None = None,
    ) -> None:
        self.predictor = predictor or UnconfiguredEmotionPredictor()
        metadata = getattr(self.predictor, "metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        self.model_version = metadata.get("model_version")
        self.calibration_status = str(metadata.get("calibration_status", "UNCALIBRATED"))
        self.rejection_threshold = metadata.get("rejection_threshold")
        sample_rate = metadata.get("sample_rate", sample_rate)
        self.sample_rate = int(sample_rate)
        self.window_s = float(window_s if window_s is not None else metadata.get("window_s", 1.5))
        self.update_interval_s = float(
            update_interval_s if update_interval_s is not None else metadata.get("update_interval_s", 0.5)
        )
        self.frame_ms = float(frame_ms if frame_ms is not None else metadata.get("frame_ms", 25.0))
        self.hop_ms = float(hop_ms if hop_ms is not None else metadata.get("hop_ms", 10.0))
        self.activity_rms_threshold = float(
            activity_rms_threshold if activity_rms_threshold is not None else metadata.get("activity_rms_threshold", 0.01)
        )
        if (
            self.sample_rate <= 1000
            or not np.isfinite(self.window_s)
            or not np.isfinite(self.update_interval_s)
            or self.window_s <= 0
            or self.update_interval_s <= 0
            or self.update_interval_s > self.window_s
            or not np.isfinite(self.frame_ms)
            or not np.isfinite(self.hop_ms)
            or self.frame_ms <= 0
            or self.hop_ms <= 0
            or self.hop_ms > self.frame_ms
            or not np.isfinite(self.activity_rms_threshold)
            or self.activity_rms_threshold < 0
        ):
            raise ValueError("model or runtime audio window metadata is invalid")
        self.window_samples = round(self.sample_rate * self.window_s)
        self.update_interval_ms = round(self.update_interval_s * 1000)
        self.buffer = AudioRingBuffer(self.window_samples)
        gates = metadata.get("quality_gates", {})
        if not isinstance(gates, dict):
            gates = {}
        self.min_activity_ratio = float(gates.get("min_activity_ratio", 0.20))
        self.min_contiguous_active_frames = int(gates.get("min_contiguous_active_frames", 3))
        self.clipping_limit = float(gates.get("clipping_limit", 0.05))
        self.minimum_snr_db = float(gates.get("minimum_snr_db", 5.0))
        self.acceptable_snr_db = float(gates.get("acceptable_snr_db", 10.0))
        self.min_audio_quality = 0.50
        self._last_update_ms: int | None = None
        self._recent_labels: deque[Emotion] = deque(maxlen=3)
        self._smoothed_emotion: Emotion | None = None

    def clear_audio_cache(self) -> None:
        """Erase the rolling audio window and short-lived label smoothing state."""
        self.buffer.clear()
        self._last_update_ms = None
        self._recent_labels.clear()
        self._smoothed_emotion = None

    def feed(self, samples: np.ndarray, timestamp_ms: int) -> EmotionOnlyResult | None:
        values = np.asarray(samples, dtype=np.float32)
        if values.ndim != 1:
            raise ValueError("experimental runtime expects mono audio samples")
        if timestamp_ms < 0:
            raise ValueError("audio timestamp must be non-negative")
        self.buffer.append(values)
        if self.buffer.size < self.window_samples:
            return None
        if self._last_update_ms is not None and timestamp_ms - self._last_update_ms < self.update_interval_ms:
            return None
        self._last_update_ms = timestamp_ms
        audio = AudioBuffer(self.buffer.snapshot(), self.sample_rate)
        quality = extract_features(
            audio,
            frame_ms=self.frame_ms,
            activity_rms_threshold=self.activity_rms_threshold,
            min_activity_ratio=self.min_activity_ratio,
            min_contiguous_active_frames=self.min_contiguous_active_frames,
            clipping_limit=self.clipping_limit,
            minimum_snr_db=self.minimum_snr_db,
            acceptable_snr_db=self.acceptable_snr_db,
        )
        if quality.active_frame_count == 0:
            return self._rejected(quality, timestamp_ms, SystemStatus.SILENCE.value)
        if (
            quality.activity_ratio < self.min_activity_ratio
            or quality.longest_active_run < self.min_contiguous_active_frames
        ):
            return self._rejected(quality, timestamp_ms, "LOW_ACTIVITY")
        if quality.quality_status is AudioQualityStatus.LOW_QUALITY:
            return self._rejected(quality, timestamp_ms, SystemStatus.LOW_QUALITY.value)
        if quality.quality_status is not AudioQualityStatus.ACCEPTABLE:
            return self._rejected(quality, timestamp_ms, "UNCERTAIN_AUDIO_QUALITY")
        if quality.audio_quality < self.min_audio_quality:
            return self._rejected(quality, timestamp_ms, SystemStatus.LOW_QUALITY.value)

        model_status = getattr(self.predictor, "model_status", ModelStatus.READY)
        if model_status is not ModelStatus.READY:
            return self._rejected(quality, timestamp_ms, model_status.value)
        try:
            emotion_features = extract_emotion_features(
                audio,
                frame_ms=self.frame_ms,
                hop_ms=self.hop_ms,
                activity_rms_threshold=self.activity_rms_threshold,
            )
            if emotion_features.valid_frame_fraction < 0.95:
                return self._rejected(quality, timestamp_ms, "INVALID_FRAME_CONTENT")
            prediction = self.predictor.predict(emotion_features, timestamp_ms=timestamp_ms)
        except Exception:
            return self._rejected(quality, timestamp_ms, "PREDICTOR_ERROR")
        all_scores = {emotion: float(prediction.scores.get(emotion, 0.0)) for emotion in Emotion}
        calibrated = (
            prediction.model_status is ModelStatus.READY
            and prediction.calibration_status == "CALIBRATED"
            and bool(prediction.model_version)
            and bool(prediction.validation_record_id)
            and prediction.decision_threshold is not None
        )
        if not calibrated:
            reason = prediction.rejection_status.value if prediction.rejection_status else "MODEL_NOT_CALIBRATED"
            self._reset_smoother()
            return self._result(
                quality, timestamp_ms, scores=all_scores, confidence=prediction.confidence,
                reason=reason, model_version=prediction.model_version,
                calibration_status=prediction.calibration_status,
                threshold=prediction.decision_threshold, emotion_features=emotion_features,
            )
        if prediction.emotion is None:
            self._reset_smoother()
            reason = prediction.rejection_status.value if prediction.rejection_status else "BELOW_VALIDATION_THRESHOLD"
            return self._result(
                quality, timestamp_ms, scores=all_scores, confidence=prediction.confidence,
                reason=reason, model_version=prediction.model_version,
                calibration_status=prediction.calibration_status,
                threshold=prediction.decision_threshold, emotion_features=emotion_features,
            )
        smoothed = self._smooth_label(prediction.emotion)
        return self._result(
            quality, timestamp_ms, scores=all_scores, confidence=prediction.confidence,
            candidate=prediction.emotion, emotion=smoothed, model_version=prediction.model_version,
            calibration_status=prediction.calibration_status,
            threshold=prediction.decision_threshold, emotion_features=emotion_features,
        )

    def _smooth_label(self, candidate: Emotion) -> Emotion:
        self._recent_labels.append(candidate)
        if self._smoothed_emotion is None:
            self._smoothed_emotion = candidate
        elif candidate is self._smoothed_emotion:
            pass
        elif sum(label is candidate for label in self._recent_labels) >= 2:
            self._smoothed_emotion = candidate
        return self._smoothed_emotion

    def _reset_smoother(self) -> None:
        self._recent_labels.clear()
        self._smoothed_emotion = None

    def _rejected(self, quality: AcousticFeatures, timestamp_ms: int, reason: str) -> EmotionOnlyResult:
        self._reset_smoother()
        return self._result(
            quality,
            timestamp_ms,
            reason=reason,
            model_version=self.model_version,
            calibration_status=self.calibration_status,
            threshold=self.rejection_threshold,
        )

    @staticmethod
    def _result(
        quality: AcousticFeatures,
        timestamp_ms: int,
        *,
        scores: Mapping[Emotion, float] | None = None,
        confidence: float = 0.0,
        reason: str | None = None,
        candidate: Emotion | None = None,
        emotion: Emotion | None = None,
        model_version: str | None = None,
        calibration_status: str = "UNCALIBRATED",
        threshold: float | None = None,
        emotion_features: EmotionFeatures | None = None,
    ) -> EmotionOnlyResult:
        return EmotionOnlyResult(
            mode=EXPERIMENTAL_MODE,
            identity_status="NOT_EVALUATED",
            emotion=emotion,
            candidate_emotion=candidate,
            confidence=float(confidence),
            scores=dict(scores or {emotion: 0.0 for emotion in Emotion}),
            rejection_reason=reason,
            audio_active=quality.active_frame_count > 0,
            activity_ratio=quality.activity_ratio,
            audio_quality=quality.audio_quality,
            audio_quality_status=quality.quality_status.value,
            timestamp_ms=timestamp_ms,
            model_version=model_version,
            calibration_status=calibration_status,
            rejection_threshold=threshold,
            audio_features=quality,
            emotion_features=emotion_features,
        )
