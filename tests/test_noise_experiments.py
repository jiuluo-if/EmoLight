import numpy as np
import pytest
import csv
import wave

from emolight.training.noise import (
    add_background_noise_snr,
    add_interferer_sir,
    add_white_noise_snr,
    apply_reverb,
    load_noise_manifest,
)


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


def test_background_music_or_fan_recording_can_be_mixed_at_requested_snr():
    target = 0.1 * np.sin(2.0 * np.pi * 220 * np.arange(8000) / 8000)
    background = 0.04 * np.sin(2.0 * np.pi * 60 * np.arange(3000) / 8000)

    for expected_snr in (20, 10, 5, 0):
        mixed = add_background_noise_snr(target, background, expected_snr)
        fitted_noise = mixed - target
        measured = 10.0 * np.log10(np.mean(target ** 2) / np.mean(fitted_noise ** 2))
        assert measured == pytest.approx(expected_snr, abs=0.2)


def test_room_impulse_response_changes_audio_without_changing_window_size():
    target = 0.1 * np.sin(2.0 * np.pi * 180 * np.arange(8000) / 8000)
    rir = np.zeros(1200)
    rir[0] = 1.0
    rir[400] = 0.4

    reverberant = apply_reverb(target, rir)

    assert reverberant.shape == target.shape
    assert np.isfinite(reverberant).all()
    assert not np.allclose(reverberant, target)


def test_noise_manifest_loads_local_music_fan_and_rir_sources(tmp_path):
    paths = {}
    for name in ("music", "fan", "rir"):
        path = tmp_path / f"{name}.wav"
        samples = (0.1 * np.sin(2 * np.pi * 220 * np.arange(8000) / 8000) * 32767).astype(np.int16)
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(8000)
            wav.writeframes(samples.tobytes())
        paths[name] = path
    manifest_path = tmp_path / "noise_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("noise_type", "path"))
        writer.writeheader()
        writer.writerows((
            {"noise_type": "background_music", "path": paths["music"].name},
            {"noise_type": "fan", "path": paths["fan"].name},
            {"noise_type": "reverb_impulse", "path": paths["rir"].name},
        ))

    sources, metadata = load_noise_manifest(manifest_path)

    assert set(sources) == {"background_music", "fan", "reverb_impulse"}
    assert all(source.sample_rate == 8000 for variants in sources.values() for source in variants)
    assert metadata["manifest_sha256"]
    assert all(len(digest) == 64 for entries in metadata["source_sha256"].values() for digest in entries)
