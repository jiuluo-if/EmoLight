import numpy as np

from emolight.audio.wav import AudioBuffer
from emolight.features.emotion_v1 import EMOTION_FEATURE_NAMES, EMOTION_FEATURE_SCHEMA_VERSION
from emolight.features.prosody import PROSODY_FEATURE_NAMES, PROSODY_SCHEMA_VERSION, extract_prosody


SAMPLE_RATE = 16000


def tone(frequency, amplitude=0.2):
    times = np.arange(SAMPLE_RATE, dtype=np.float64) / SAMPLE_RATE
    return (amplitude * np.sin(2.0 * np.pi * frequency * times)).astype(np.float32)


def test_public_prosody_api_uses_versioned_24_feature_contract():
    features = extract_prosody(AudioBuffer(tone(220.0), SAMPLE_RATE))

    assert PROSODY_SCHEMA_VERSION == EMOTION_FEATURE_SCHEMA_VERSION == "emotion-prosody-24-v1"
    assert PROSODY_FEATURE_NAMES == EMOTION_FEATURE_NAMES
    assert features.schema_version == PROSODY_SCHEMA_VERSION
    assert features.values.shape == (24,)
    assert abs(features.values[0] - 220.0) < 8.0
    assert features.valid_frame_fraction > 0.95


def test_prosody_missing_f0_is_nan_and_does_not_become_true_zero_hz():
    features = extract_prosody(AudioBuffer(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE))

    assert np.isnan(features.values[:7]).all()
    assert features.values[7] == 0.0
    assert features.f0_valid_fraction == 0.0
    assert np.isfinite(features.values[7:23]).all()


def test_valid_frame_fraction_detects_nan_inf_and_incomplete_windows():
    samples = tone(180.0)
    samples[4000] = np.nan
    samples[9000] = np.inf

    features = extract_prosody(AudioBuffer(samples, SAMPLE_RATE))

    assert features.valid_frame_fraction < 1.0
    assert np.isfinite(features.simple_values).all()
    assert np.isfinite(features.values[7:23]).all()


def test_energy_activity_and_periodicity_are_not_voice_or_identity_verdicts():
    times = np.arange(SAMPLE_RATE, dtype=np.float64) / SAMPLE_RATE
    low_tone = (0.004 * np.sin(2.0 * np.pi * 260.0 * times)).astype(np.float32)

    features = extract_prosody(AudioBuffer(low_tone, SAMPLE_RATE))

    assert features.values[7] > 0.9
    assert features.active_fraction == 0.0
    assert features.mean_periodicity > 0.5
    assert not hasattr(features, "speech_confirmed")
    assert not hasattr(features, "target_identity")
