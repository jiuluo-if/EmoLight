from collections import deque

import numpy as np

from emolight.emotion.predictor import EmotionPrediction, ModelStatus
from emolight.events import Emotion, SystemStatus
from emolight.runtime.experimental import EmotionOnlyExperimentalRuntime


class SequencePredictor:
    model_status = ModelStatus.READY
    metadata = {
        "model_version": "test-trained-model",
        "calibration_status": "CALIBRATED",
        "rejection_threshold": 0.6,
    }

    def __init__(self, labels):
        self.labels = deque(labels)
        self.calls = 0

    def predict(self, features, timestamp_ms=0):
        self.calls += 1
        emotion = self.labels.popleft()
        scores = {candidate: (0.8 if candidate is emotion else 0.2 / 3) for candidate in Emotion}
        return EmotionPrediction(
            emotion=emotion,
            confidence=scores[emotion],
            scores=scores,
            model_status=ModelStatus.READY,
            timestamp_ms=timestamp_ms,
            model_version="test-trained-model",
            calibration_status="CALIBRATED",
            decision_threshold=0.6,
            validation_record_id="v" * 64,
        )


def clean_window():
    rate = 16000
    rng = np.random.default_rng(41)
    samples = rng.normal(0.0, 0.002, round(rate * 1.5)).astype(np.float32)
    times = np.arange(samples.size - rate // 3) / rate
    samples[rate // 3:] += (0.12 * np.sin(2 * np.pi * 220 * times)).astype(np.float32)
    return samples


def test_experimental_runtime_emits_scores_without_target_identity_event_and_smooths_labels():
    predictor = SequencePredictor((Emotion.HAPPY, Emotion.ANGRY, Emotion.ANGRY))
    runtime = EmotionOnlyExperimentalRuntime(predictor)

    results = [runtime.feed(clean_window(), timestamp_ms) for timestamp_ms in (1500, 2000, 2500)]

    assert all(result is not None for result in results)
    assert results[0].mode == "EMOTION_ONLY_EXPERIMENTAL"
    assert results[0].identity_status == "NOT_EVALUATED"
    assert results[0].timestamp_ms == 1500
    assert set(results[0].scores) == set(Emotion)
    assert results[0].candidate_emotion is Emotion.HAPPY
    assert results[1].candidate_emotion is Emotion.ANGRY
    assert results[1].emotion is Emotion.HAPPY
    assert results[2].emotion is Emotion.ANGRY
    assert predictor.calls == 3
    assert not hasattr(results[0], "target_event")


def test_experimental_runtime_rejects_silence_and_strong_noise_before_classification():
    predictor = SequencePredictor((Emotion.HAPPY,))
    runtime = EmotionOnlyExperimentalRuntime(predictor)

    silence = runtime.feed(np.zeros(24000, dtype=np.float32), timestamp_ms=1500)
    noise = np.random.default_rng(5).normal(0.0, 0.15, 24000).astype(np.float32)
    rejected_noise = runtime.feed(noise, timestamp_ms=2000)

    assert silence.rejection_reason == SystemStatus.SILENCE.value
    assert silence.calibration_status == "CALIBRATED"
    assert silence.rejection_threshold == 0.6
    assert rejected_noise.rejection_reason == "UNCERTAIN_AUDIO_QUALITY"
    assert rejected_noise.emotion is None
    assert predictor.calls == 0
