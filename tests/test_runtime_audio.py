import numpy as np

from emolight.audio.ring_buffer import AudioRingBuffer
from emolight.runtime.pipeline import RealtimeFeatureRuntime
from emolight.events import SystemStatus


def test_ring_buffer_keeps_only_latest_samples_in_chronological_order():
    buffer = AudioRingBuffer(capacity_samples=5)
    buffer.append(np.array([0, 1, 2], dtype=np.float32))
    buffer.append(np.array([3, 4, 5, 6], dtype=np.float32))

    assert buffer.size == 5
    assert buffer.snapshot().tolist() == [2, 3, 4, 5, 6]


def test_realtime_features_use_bounded_window_and_emit_not_configured():
    runtime = RealtimeFeatureRuntime(sample_rate=1000, window_s=1, update_interval_s=0.5)
    background = np.full(250, 0.001, dtype=np.float32)
    chunk = np.full(250, 0.1, dtype=np.float32)

    assert runtime.feed(background, timestamp_ms=250) is None
    assert runtime.feed(background, timestamp_ms=500) is None
    assert runtime.feed(chunk, timestamp_ms=750) is None
    first = runtime.feed(chunk, timestamp_ms=1000)
    assert first is not None and first.status is SystemStatus.NOT_CONFIGURED
    assert runtime.feed(chunk, timestamp_ms=1250) is None
    second = runtime.feed(chunk, timestamp_ms=1500)
    assert second is not None and second.emotion is None


def test_realtime_silence_is_rejected_before_model_prediction():
    runtime = RealtimeFeatureRuntime(sample_rate=1000, window_s=1, update_interval_s=0.5)
    event = runtime.feed(np.zeros(1000, dtype=np.float32), timestamp_ms=1000)

    assert event is not None
    assert event.status is SystemStatus.SILENCE
    assert event.emotion is None


def test_runtime_snapshot_keeps_audio_quality_and_event_on_the_same_window():
    from emolight.emotion.predictor import EmotionPrediction, ModelStatus
    from emolight.events import Emotion, SystemStatus
    from emolight.speaker.verifier import SpeakerVerification

    class HappyPredictor:
        def predict(self, features, timestamp_ms=0):
            return EmotionPrediction(Emotion.HAPPY, 0.9, model_status=ModelStatus.READY)

    class TargetVerifier:
        def verify(self, audio):
            return SpeakerVerification(SystemStatus.TARGET_ACTIVE, 0.95)

    samples = np.full(1000, 0.001, dtype=np.float32)
    times = np.arange(500, dtype=np.float32) / 1000
    samples[500:] += 0.1 * np.sin(2 * np.pi * 120 * times)
    runtime = RealtimeFeatureRuntime(
        sample_rate=1000,
        window_s=1,
        update_interval_s=0.5,
        predictor=HappyPredictor(),
        speaker_verifier=TargetVerifier(),
        frame_ms=20.0,
        hop_ms=5.0,
    )

    snapshot = runtime.feed_snapshot(samples, timestamp_ms=1000)

    assert snapshot is not None
    assert snapshot.acoustic_features is runtime.latest_features
    assert snapshot.event.timestamp_ms == snapshot.timestamp_ms == 1000
    assert snapshot.event.audio_quality == snapshot.acoustic_features.audio_quality
    assert snapshot.prosody_features.frame_ms == 20.0
    assert snapshot.prosody_features.hop_ms == 5.0
