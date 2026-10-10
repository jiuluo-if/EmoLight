import io
import wave

import numpy as np

from emolight.events import Emotion
from emolight.runtime.experimental import EmotionOnlyExperimentalRuntime
from emolight.training.live_emotion_only import LatencySamples, _display_provenance, _latency, _publish_latest, _result_payload, replay_wav
from test_experimental_runtime import SequencePredictor, clean_window


def _write_wav(path, samples, sample_rate=16000):
    pcm = np.clip(samples * 32767, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm.tobytes())


def test_wav_replay_uses_same_experimental_window_path_and_emits_four_scores(tmp_path):
    samples = np.concatenate((clean_window(), clean_window()[:8000]))
    wav_path = tmp_path / "emodb_clip.wav"
    _write_wav(wav_path, samples)
    predictor = SequencePredictor((Emotion.HAPPY, Emotion.HAPPY))
    runtime = EmotionOnlyExperimentalRuntime(predictor)

    results, summary = replay_wav(wav_path, runtime, frame_ms=20)

    assert len(results) == 2
    assert all(result.mode == "EMOTION_ONLY_EXPERIMENTAL" for result in results)
    assert all(result.identity_status == "NOT_EVALUATED" for result in results)
    assert all(set(result.scores) == set(Emotion) for result in results)
    assert all(result.emotion is Emotion.HAPPY for result in results)
    assert _result_payload(results[0])["confidence_for"] == "happy"
    assert results[1].timestamp_ms - results[0].timestamp_ms >= 500
    assert summary["window_updates"] == 2
    assert summary["audio_duration_s"] == 2.0
    assert summary["processing_ms"]["per_window_update"]["p95"] >= 0.0


def test_latency_samples_remain_bounded_and_report_total_count():
    samples = LatencySamples(capacity=3)
    for value in range(100):
        samples.add(float(value))

    summary = _latency(samples)

    assert samples.retained_sample_count == 3
    assert summary["sample_count"] == 100
    assert summary["retained_sample_count"] == 3
    assert summary["median"] == 98.0
    assert summary["p95"] == 98.9


def test_latest_display_queue_keeps_only_the_newest_result():
    import queue

    results = queue.Queue(maxsize=1)

    assert _publish_latest(results, {"timestamp_ms": 1}) is False
    assert _publish_latest(results, {"timestamp_ms": 2}) is True
    assert results.qsize() == 1
    assert results.get_nowait() == {"timestamp_ms": 2}


def test_live_session_provenance_omits_speaker_identifier_lists():
    display = _display_provenance({
        "dataset_id": "example",
        "split": {"split_speaker_ids": {"train": ["private-id"]}},
    })

    assert display == {"dataset_id": "example", "split": {}}
    assert "private-id" not in repr(display)
