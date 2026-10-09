import numpy as np


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


def _finite_mono(samples: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(samples, dtype=np.float64)
    if result.ndim != 1:
        raise ValueError(f"{name} must be a mono vector")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result
