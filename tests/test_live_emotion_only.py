import io
import wave

import numpy as np

from emolight.events import Emotion
from emolight.runtime.experimental import EmotionOnlyExperimentalRuntime
from emolight.training.live_emotion_only import _result_payload, replay_wav
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
