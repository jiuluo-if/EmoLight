import numpy as np

from emolight.audio.wav import AudioBuffer
from emolight.features.prosody import (
    PROSODY_FEATURE_NAMES,
    PROSODY_SCHEMA_VERSION,
    extract_prosody,
    _energy_summary,
)


SAMPLE_RATE = 16000


def tone(frequency, duration=1.0, amplitude=0.2):
    times = np.arange(round(SAMPLE_RATE * duration), dtype=np.float64) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * frequency * times)).astype(np.float32)


def test_prosody_vector_has_fixed_versioned_schema_and_recovers_f0():
    features = extract_prosody(AudioBuffer(tone(220), SAMPLE_RATE))

    assert features.values.shape == (32,)
    assert features.feature_names == PROSODY_FEATURE_NAMES
    assert features.schema_version == PROSODY_SCHEMA_VERSION == "prosody-v1"
    assert features.sample_rate == SAMPLE_RATE
    assert features.frame_length_samples == 400
    assert features.hop_length_samples == 160
    assert features.voiced_fraction > 0.9
    assert abs(features.values[0] - 220) < 8
    assert np.isfinite(features.values).all()


def test_prosody_tracks_a_past_to_current_pitch_step():
    samples = np.concatenate((tone(180, 0.5), tone(280, 0.5)))
    features = extract_prosody(AudioBuffer(samples, SAMPLE_RATE))

    assert features.values[9] > 60
    assert features.values[12] > 0


def test_silence_has_no_pitch_and_marks_pauses():
    features = extract_prosody(AudioBuffer(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE))

    assert features.values[0] == 0.0
    assert features.voiced_fraction == 0.0
    assert features.values[25] == 1.0
    assert features.values[26] == 0.0
    assert np.isfinite(features.values).all()


def test_noise_and_nonfinite_samples_never_create_nonfinite_features():
    rng = np.random.default_rng(8)
    noisy = tone(200, amplitude=0.1) + rng.normal(0.0, 0.02, SAMPLE_RATE).astype(np.float32)
    noisy[10] = np.nan
    noisy[20] = np.inf
    noisy[30] = -np.inf
    features = extract_prosody(AudioBuffer(noisy, SAMPLE_RATE))

    assert features.values.shape == (32,)
    assert np.isfinite(features.values).all()
    assert 0.0 <= features.voiced_fraction <= 1.0


def test_short_input_is_padded_only_to_the_current_frame_boundary():
    features = extract_prosody(AudioBuffer(tone(250, duration=0.01), SAMPLE_RATE))

    assert features.frame_count == 1
    assert features.values.shape == (32,)
    assert features.voiced_fraction == 0.0
    assert np.isfinite(features.values).all()


def test_energy_delta_and_slope_values_follow_the_published_feature_names():
    rms = np.asarray((1.0, 2.0, 4.0, 8.0))
    times = np.asarray((0.0, 1.0, 2.0, 3.0))
    deltas = np.diff(rms)

    summary = _energy_summary(rms, deltas, times)

    assert summary[8] == np.mean(deltas)
    assert summary[9] == np.std(deltas)
    assert summary[10] == np.polyfit(times, rms, 1)[0]
