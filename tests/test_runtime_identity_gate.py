import numpy as np
import pytest

from emolight.emotion.predictor import EmotionPrediction, ModelStatus
from emolight.events import Emotion, EmotionEvent, SystemStatus
from emolight.runtime.pipeline import RealtimeFeatureRuntime
from emolight.speaker.verifier import SpeakerVerification


class AlwaysHappyPredictor:
    def __init__(self):
        self.calls = 0

    def predict(self, features, timestamp_ms=0):
        self.calls += 1
        return EmotionPrediction(Emotion.HAPPY, 1.0, model_status=ModelStatus.READY)


class FixedSpeakerVerifier:
    def __init__(self, status, confidence=0.99):
        self.result = SpeakerVerification(status, confidence)
        self.calls = 0

    def verify(self, audio):
        self.calls += 1
        return self.result


def runtime_for(verifier=None, predictor=None):
    return RealtimeFeatureRuntime(
        sample_rate=1000,
        window_s=1,
        update_interval_s=0.5,
        speaker_verifier=verifier,
        predictor=predictor,
    )


def active_window():
    samples = np.full(1000, 0.001, dtype=np.float32)
    times = np.arange(500, dtype=np.float32) / 1000
    samples[500:] += 0.1 * np.sin(2 * np.pi * 120 * times)
    return samples


def test_unconfigured_identity_blocks_high_confidence_emotion_predictor():
    predictor = AlwaysHappyPredictor()
    runtime = runtime_for(predictor=predictor)

    event = runtime.feed(active_window(), timestamp_ms=1000)

    assert event is not None
    assert event.status is SystemStatus.NOT_CONFIGURED
    assert event.emotion is None
    assert predictor.calls == 0


def test_emotion_prediction_cannot_encode_an_identity_status():
    with pytest.raises(ValueError):
        EmotionPrediction(
            emotion=None,
            model_status=ModelStatus.READY,
            rejection_status=SystemStatus.TARGET_ACTIVE,
        )


@pytest.mark.parametrize("status", [SystemStatus.NON_TARGET, SystemStatus.OVERLAP, SystemStatus.NOT_CONFIGURED])
def test_non_target_overlap_and_unconfigured_identity_never_call_predictor(status):
    predictor = AlwaysHappyPredictor()
    verifier = FixedSpeakerVerifier(status)
    runtime = runtime_for(verifier, predictor)

    event = runtime.feed(active_window(), timestamp_ms=1000)

    assert event is not None and event.status is status
    assert event.emotion is None
    assert predictor.calls == 0


def test_only_verified_target_identity_can_create_target_emotion_event():
    predictor = AlwaysHappyPredictor()
    verifier = FixedSpeakerVerifier(SystemStatus.TARGET_ACTIVE, confidence=0.96)
    runtime = runtime_for(verifier, predictor)

    event = runtime.feed(active_window(), timestamp_ms=1000)

    assert event is not None
    assert event.status is SystemStatus.TARGET_ACTIVE
    assert event.emotion is Emotion.HAPPY
    assert event.emotion_confidence == 1.0
    assert event.speaker_confidence == 0.96
    assert predictor.calls == 1


def test_low_speaker_confidence_is_rejected_before_emotion_prediction():
    predictor = AlwaysHappyPredictor()
    verifier = FixedSpeakerVerifier(SystemStatus.TARGET_ACTIVE, confidence=0.2)

    event = runtime_for(verifier, predictor).feed(active_window(), timestamp_ms=1000)

    assert event is not None and event.status is SystemStatus.UNCERTAIN
    assert event.emotion is None
    assert predictor.calls == 0


def test_silence_and_one_frame_transient_do_not_reach_speaker_or_emotion_models():
    predictor = AlwaysHappyPredictor()
    verifier = FixedSpeakerVerifier(SystemStatus.TARGET_ACTIVE)

    silence = runtime_for(verifier, predictor).feed(np.zeros(1000, dtype=np.float32), timestamp_ms=1000)
    transient_samples = np.zeros(1000, dtype=np.float32)
    transient_samples[300] = 1.0
    transient_runtime = runtime_for(verifier, predictor)
    transient = transient_runtime.feed(transient_samples, timestamp_ms=1000)

    assert silence is not None and silence.status is SystemStatus.SILENCE
    assert transient is not None and transient.status is SystemStatus.UNCERTAIN
    assert verifier.calls == 0
    assert predictor.calls == 0
