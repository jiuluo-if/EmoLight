import wave

import numpy as np

from emolight.audio.wav import load_wav
from emolight.emotion.predictor import UnconfiguredEmotionPredictor
from emolight.features.acoustic import extract_features
from emolight.events import SystemStatus


def write_wav(path, samples, sample_rate=16000):
    pcm = np.asarray(samples, dtype=np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def test_wav_features_are_acoustic_and_model_stays_unconfigured(tmp_path):
    times = np.arange(16000) / 16000
    samples = (np.sin(2 * np.pi * 220 * times) * 10000).astype(np.int16)
    path = tmp_path / "tone.wav"
    write_wav(path, samples)

    audio = load_wav(path)
    features = extract_features(audio)
    event = UnconfiguredEmotionPredictor().predict(features, timestamp_ms=700)

    assert audio.sample_rate == 16000
    assert features.duration_s == 1.0
    assert 0.0 < features.rms < 1.0
    assert 0.0 <= features.zero_crossing_rate <= 1.0
    assert event.status is SystemStatus.NOT_CONFIGURED
    assert event.emotion is None


def test_silence_is_marked_low_quality(tmp_path):
    path = tmp_path / "silence.wav"
    write_wav(path, np.zeros(8000, dtype=np.int16))

    features = extract_features(load_wav(path))
    event = UnconfiguredEmotionPredictor().predict(features)

    assert features.audio_quality < 0.5
    assert event.status is SystemStatus.LOW_QUALITY
