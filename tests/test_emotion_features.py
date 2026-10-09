import numpy as np
import pytest

from emolight.audio.wav import AudioBuffer
from emolight.features.emotion_v1 import (
    EMOTION_FEATURE_NAMES,
    EMOTION_FEATURE_SCHEMA_VERSION,
    SIMPLE_FEATURE_NAMES,
    extract_emotion_features,
)
from emolight.training.noise import add_white_noise_snr


SAMPLE_RATE = 16000


def tone(f0_hz: float, *, amplitude: float = 0.15, duration_s: float = 1.0) -> np.ndarray:
    times = np.arange(round(SAMPLE_RATE * duration_s), dtype=np.float64) / SAMPLE_RATE
    return (amplitude * np.sin(2.0 * np.pi * f0_hz * times)).astype(np.float32)


def test_full_and_simple_feature_vectors_have_versioned_fixed_schemas():
    features = extract_emotion_features(AudioBuffer(tone(220), SAMPLE_RATE))

    assert features.schema_version == EMOTION_FEATURE_SCHEMA_VERSION
    assert features.schema_version == "emotion-prosody-24-v1"
    assert features.values.shape == (24,)
    assert features.feature_names == EMOTION_FEATURE_NAMES
    assert features.simple_values.shape == (len(SIMPLE_FEATURE_NAMES),)
    assert len(EMOTION_FEATURE_NAMES) == 24
    assert features.valid_frame_fraction > 0.95


def test_f0_is_missing_not_zero_for_silence_noise_and_invalid_frames():
    silence = extract_emotion_features(AudioBuffer(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE))
    rng = np.random.default_rng(14)
    noise = extract_emotion_features(AudioBuffer(rng.normal(0.0, 0.1, SAMPLE_RATE).astype(np.float32), SAMPLE_RATE))
    corrupted_audio = tone(220)
    corrupted_audio[4000] = np.nan
    corrupted_audio[9000] = np.inf
    corrupted = extract_emotion_features(AudioBuffer(corrupted_audio, SAMPLE_RATE))

    assert np.isnan(silence.values[:7]).all()
    assert silence.values[7] == 0.0
    assert np.isnan(noise.values[:7]).all()
    assert noise.values[7] == 0.0
    assert corrupted.valid_frame_fraction < 1.0
    assert np.isfinite(corrupted.simple_values).all()
    assert np.isfinite(corrupted.values[7:]).all()


def test_f0_recovers_male_and_female_ranges_and_low_amplitude_periodic_voice():
    male = extract_emotion_features(AudioBuffer(tone(110.0, amplitude=0.02), SAMPLE_RATE))
    female = extract_emotion_features(AudioBuffer(tone(260.0, amplitude=0.004), SAMPLE_RATE))

    assert abs(male.values[0] - 110.0) < 8.0
    assert abs(female.values[0] - 260.0) < 12.0
    assert male.values[7] > 0.9
    assert female.values[7] > 0.9
    assert female.active_fraction == 0.0


def test_snr_noise_lowers_pitch_validity_without_becoming_zero_hz():
    speech = tone(210.0, amplitude=0.08)
    rng = np.random.default_rng(88)
    noise = rng.normal(0.0, 1.0, speech.size).astype(np.float32)

    clean = extract_emotion_features(AudioBuffer(speech, SAMPLE_RATE))
    noisy = extract_emotion_features(AudioBuffer(speech + 0.08 * noise, SAMPLE_RATE))

    assert clean.values[7] > noisy.values[7]
    assert np.isnan(noisy.values[:7]).all() or noisy.values[7] > 0.0


@pytest.mark.parametrize("snr_db", (20.0, 10.0, 5.0, 0.0))
def test_features_remain_finite_or_explicitly_missing_at_required_snrs(snr_db):
    speech = tone(220.0, amplitude=0.12)
    noisy = add_white_noise_snr(speech, snr_db, seed=123)

    features = extract_emotion_features(AudioBuffer(noisy, SAMPLE_RATE))

    assert features.valid_frame_fraction > 0.95
    assert 0.0 <= features.f0_valid_fraction <= 1.0
    assert np.isfinite(features.simple_values).all()
    assert np.isfinite(features.values[7:23]).all()
    if features.f0_valid_fraction == 0.0:
        assert np.isnan(features.values[:7]).all()
    else:
        assert 60.0 <= features.values[0] <= 500.0


def test_clipped_tone_retains_periodicity_but_does_not_claim_speech_or_identity():
    clipped = np.clip(tone(180.0), -0.03, 0.03)

    features = extract_emotion_features(AudioBuffer(clipped, SAMPLE_RATE))

    assert features.mean_periodicity > 0.5
    assert features.values[7] > 0.9
    assert not hasattr(features, "speech_confirmed")
    assert not hasattr(features, "target_identity")


def test_energy_activity_periodicity_and_identity_are_distinct_signals():
    features = extract_emotion_features(AudioBuffer(tone(200), SAMPLE_RATE))

    assert features.active_fraction > 0.95
    assert features.f0_valid_fraction > 0.95
    assert features.mean_periodicity > 0.5
    assert not hasattr(features, "is_speech")
    assert not hasattr(features, "target_identity")
