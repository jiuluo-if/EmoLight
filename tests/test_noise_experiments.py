import numpy as np
import pytest

from emolight.training.noise import add_interferer_sir, add_white_noise_snr


def test_white_noise_mixer_hits_requested_snr_without_clipping():
    signal = 0.1 * np.sin(2 * np.pi * 200 * np.arange(16000) / 16000)

    for expected_snr in (20, 10, 5, 0):
        mixed = add_white_noise_snr(signal, expected_snr, seed=7)
        noise = mixed - signal
        measured = 10 * np.log10(np.mean(signal ** 2) / np.mean(noise ** 2))
        assert measured == pytest.approx(expected_snr, abs=0.2)
        assert np.max(np.abs(mixed)) < 1.0


def test_interferer_mixer_controls_other_voice_ratio_separately():
    target = 0.1 * np.sin(2 * np.pi * 180 * np.arange(8000) / 8000)
    interferer = 0.2 * np.sin(2 * np.pi * 260 * np.arange(3000) / 8000)

    for expected_sir in (-6, 0, 6):
        mixed = add_interferer_sir(target, interferer, expected_sir)
        fitted_interferer = mixed - target
        measured = 20 * np.log10(np.sqrt(np.mean(target ** 2)) / np.sqrt(np.mean(fitted_interferer ** 2)))
        assert measured == pytest.approx(expected_sir, abs=0.2)


def test_mixers_reject_empty_or_silent_reference_signals():
    with pytest.raises(ValueError):
        add_white_noise_snr(np.zeros(8), 10, seed=1)
    with pytest.raises(ValueError):
        add_interferer_sir(np.ones(8), np.zeros(4), 0)
