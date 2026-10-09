import csv
import hashlib
from pathlib import Path

import numpy as np

from emolight.audio.wav import AudioBuffer, load_wav


def load_noise_manifest(path: str | Path) -> tuple[dict[str, list[AudioBuffer]], dict]:
    """Load local recorded backgrounds and room impulse responses with hashes."""
    manifest_path = Path(path).resolve()
    allowed_types = {"background_music", "fan", "environment", "reverb_impulse"}
    try:
        manifest_bytes = manifest_path.read_bytes()
        stream = manifest_path.open("r", encoding="utf-8-sig", newline="")
    except OSError as error:
        raise ValueError(f"cannot open noise manifest: {error}") from error
    sources: dict[str, list[AudioBuffer]] = {}
    source_hashes: dict[str, list[str]] = {}
    with stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not {"noise_type", "path"}.issubset(reader.fieldnames):
            raise ValueError("noise manifest requires noise_type,path columns")
        for row_number, row in enumerate(reader, start=2):
            noise_type = (row.get("noise_type") or "").strip().casefold()
            if noise_type not in allowed_types:
                raise ValueError(f"noise manifest row {row_number}: unsupported noise_type {noise_type!r}")
            raw_path = (row.get("path") or "").strip()
            if not raw_path:
                raise ValueError(f"noise manifest row {row_number}: path is empty")
            source_path = Path(raw_path)
            if not source_path.is_absolute():
                source_path = manifest_path.parent / source_path
            source_path = source_path.resolve()
            if not source_path.is_file():
                raise ValueError(f"noise manifest row {row_number}: file does not exist")
            sources.setdefault(noise_type, []).append(load_wav(source_path))
            source_hashes.setdefault(noise_type, []).append(_sha256_file(source_path))
    return sources, {
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_sha256": source_hashes,
        "source_counts": {kind: len(items) for kind, items in sources.items()},
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_white_noise_snr(signal: np.ndarray, snr_db: float, *, seed: int | None = None) -> np.ndarray:
    """Mix white noise at a requested signal/noise power ratio without clipping."""
    target = _finite_mono(signal, "signal")
    if target.size == 0 or float(np.mean(np.square(target))) <= 1e-12:
        raise ValueError("signal must contain non-zero energy")
    if not np.isfinite(snr_db) or not -20.0 <= snr_db <= 60.0:
        raise ValueError("snr_db must be finite and within [-20, 60]")
    noise = np.random.default_rng(seed).normal(0.0, 1.0, size=target.size)
    noise_power = float(np.mean(np.square(noise)))
    target_power = float(np.mean(np.square(target)))
    wanted_noise_power = target_power / (10.0 ** (snr_db / 10.0))
    scaled_noise = noise * np.sqrt(wanted_noise_power / noise_power)
    return np.asarray(target + scaled_noise, dtype=np.float32)


def add_background_noise_snr(signal: np.ndarray, background: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix a recorded music/fan/environment source at an explicit SNR."""
    target = _finite_mono(signal, "signal")
    noise = _finite_mono(background, "background")
    if target.size == 0 or noise.size == 0:
        raise ValueError("signal and background must be non-empty")
    if not np.isfinite(snr_db) or not -20.0 <= snr_db <= 60.0:
        raise ValueError("snr_db must be finite and within [-20, 60]")
    target_power = float(np.mean(np.square(target)))
    noise_power = float(np.mean(np.square(noise)))
    if target_power <= 1e-12 or noise_power <= 1e-12:
        raise ValueError("signal and background must contain non-zero energy")
    repeats = int(np.ceil(target.size / noise.size))
    aligned = np.tile(noise, repeats)[:target.size]
    wanted_noise_power = target_power / (10.0 ** (snr_db / 10.0))
    aligned_power = float(np.mean(np.square(aligned)))
    scaled = aligned * np.sqrt(wanted_noise_power / aligned_power)
    return np.asarray(target + scaled, dtype=np.float32)


def add_interferer_sir(target: np.ndarray, interferer: np.ndarray, sir_db: float) -> np.ndarray:
    """Mix a second recording at a target/interferer ratio, reported separately from SNR."""
    target_samples = _finite_mono(target, "target")
    interfering_samples = _finite_mono(interferer, "interferer")
    if target_samples.size == 0 or interfering_samples.size == 0:
        raise ValueError("target and interferer must be non-empty")
    if not np.isfinite(sir_db) or not -30.0 <= sir_db <= 30.0:
        raise ValueError("sir_db must be finite and within [-30, 30]")
    target_rms = float(np.sqrt(np.mean(np.square(target_samples))))
    interferer_rms = float(np.sqrt(np.mean(np.square(interfering_samples))))
    if target_rms <= 1e-8 or interferer_rms <= 1e-8:
        raise ValueError("target and interferer must both have non-zero energy")
    repeats = int(np.ceil(target_samples.size / interfering_samples.size))
    aligned = np.tile(interfering_samples, repeats)[:target_samples.size]
    aligned_rms = float(np.sqrt(np.mean(np.square(aligned))))
    scale = target_rms / (aligned_rms * 10.0 ** (sir_db / 20.0))
    return np.asarray(target_samples + aligned * scale, dtype=np.float32)


def apply_reverb(signal: np.ndarray, impulse_response: np.ndarray) -> np.ndarray:
    """Apply a recorded room impulse response and preserve the original window length."""
    target = _finite_mono(signal, "signal")
    impulse = _finite_mono(impulse_response, "impulse_response")
    if target.size == 0 or impulse.size == 0:
        raise ValueError("signal and impulse response must be non-empty")
    input_rms = float(np.sqrt(np.mean(np.square(target))))
    if input_rms <= 1e-8 or float(np.sum(np.square(impulse))) <= 1e-12:
        raise ValueError("signal and impulse response must contain non-zero energy")
    convolution_length = target.size + impulse.size - 1
    fft_size = 1 << (convolution_length - 1).bit_length()
    convolved = np.fft.irfft(
        np.fft.rfft(target, n=fft_size) * np.fft.rfft(impulse, n=fft_size),
        n=fft_size,
    )[:target.size]
    output_rms = float(np.sqrt(np.mean(np.square(convolved))))
    if output_rms <= 1e-12 or not np.isfinite(output_rms):
        raise ValueError("impulse response produced a zero or invalid output")
    return np.asarray(convolved * (input_rms / output_rms), dtype=np.float32)


def _finite_mono(samples: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(samples, dtype=np.float64)
    if result.ndim != 1:
        raise ValueError(f"{name} must be a mono vector")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result
