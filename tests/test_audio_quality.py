import numpy as np

from emolight.audio.wav import AudioBuffer
from emolight.features.acoustic import AudioQualityStatus, extract_features


SAMPLE_RATE = 16000


def test_snr_quality_uses_inactive_noise_floor_and_separates_level():
    samples = np.full(SAMPLE_RATE, 0.001, dtype=np.float32)
    times = np.arange(SAMPLE_RATE // 2) / SAMPLE_RATE
    samples[SAMPLE_RATE // 2:] += (0.1 * np.sin(2 * np.pi * 220 * times)).astype(np.float32)

    features = extract_features(AudioBuffer(samples, SAMPLE_RATE))

    assert features.rms > 0.01
    assert features.activity_ratio == 0.5
    assert features.estimated_snr_db is not None and features.estimated_snr_db > 10
    assert features.quality_status is AudioQualityStatus.ACCEPTABLE


def test_continuous_loud_noise_is_uncertain_without_noise_floor_estimate():
    noise = np.random.default_rng(11).normal(0.0, 0.2, SAMPLE_RATE).astype(np.float32)

    features = extract_features(AudioBuffer(noise, SAMPLE_RATE))

    assert features.rms > 0.1
    assert features.activity_ratio == 1.0
    assert features.estimated_snr_db is None
    assert features.quality_status is AudioQualityStatus.UNCERTAIN


def test_low_snr_and_clipping_are_distinct_quality_failures():
    low_snr = np.full(SAMPLE_RATE, 0.008, dtype=np.float32)
    times = np.arange(SAMPLE_RATE // 2) / SAMPLE_RATE
    low_snr[SAMPLE_RATE // 2:] += (0.01 * np.sin(2 * np.pi * 180 * times)).astype(np.float32)
    low_features = extract_features(AudioBuffer(low_snr, SAMPLE_RATE))
    clipped = np.full(SAMPLE_RATE, 1.0, dtype=np.float32)
    clip_features = extract_features(AudioBuffer(clipped, SAMPLE_RATE))

    assert low_features.estimated_snr_db is not None and low_features.estimated_snr_db < 0
    assert low_features.quality_status is AudioQualityStatus.LOW_QUALITY
    assert clip_features.clipping_fraction == 1.0
    assert clip_features.quality_status is AudioQualityStatus.LOW_QUALITY
