from dataclasses import dataclass

import numpy as np

from emolight.audio.wav import AudioBuffer


PROSODY_SCHEMA_VERSION = "prosody-v1"
PROSODY_FEATURE_NAMES = (
    "f0_median_hz", "f0_mean_hz", "f0_std_hz", "f0_min_hz", "f0_q10_hz",
    "f0_q25_hz", "f0_q75_hz", "f0_q90_hz", "f0_max_hz", "f0_range_hz",
    "f0_delta_mean_hz_s", "f0_delta_std_hz_s", "f0_slope_hz_s",
    "rms_mean", "rms_std", "rms_q10", "rms_median", "rms_q90", "rms_range",
    "log_rms_mean", "log_rms_std", "rms_delta_mean_s", "rms_delta_std_s", "rms_slope_s",
    "voiced_ratio", "pause_ratio", "activity_ratio", "longest_voiced_ratio",
    "zcr_mean", "zcr_std", "zcr_q90", "hnr_median_db",
)


@dataclass(frozen=True)
class ProsodyFeatures:
    values: np.ndarray
    feature_names: tuple[str, ...]
    schema_version: str
    sample_rate: int
    frame_ms: float
    hop_ms: float
    frame_length_samples: int
    hop_length_samples: int
    frame_count: int
    voiced_fraction: float

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=np.float32, copy=True)
        if values.shape != (len(PROSODY_FEATURE_NAMES),):
            raise ValueError(f"prosody values must have {len(PROSODY_FEATURE_NAMES)} entries")
        if not np.isfinite(values).all():
            raise ValueError("prosody values must be finite")
        if self.feature_names != PROSODY_FEATURE_NAMES or self.schema_version != PROSODY_SCHEMA_VERSION:
            raise ValueError("prosody feature schema mismatch")
        if self.sample_rate <= 0 or self.frame_length_samples <= 0 or self.hop_length_samples <= 0:
            raise ValueError("prosody sampling metadata must be positive")
        if not 0.0 <= self.voiced_fraction <= 1.0 or self.frame_count < 0:
            raise ValueError("prosody frame metadata is invalid")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)


def extract_prosody(
    audio: AudioBuffer,
    *,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    min_f0_hz: float = 60.0,
    max_f0_hz: float = 500.0,
    activity_rms_threshold: float = 0.01,
) -> ProsodyFeatures:
    """Extract a fixed causal-window prosody vector; no future audio or text is used."""
    samples = np.asarray(audio.samples, dtype=np.float32)
    if samples.ndim != 1 or audio.sample_rate <= 0:
        raise ValueError("audio must be mono with a positive sample rate")
    if frame_ms <= 0 or hop_ms <= 0 or hop_ms > frame_ms:
        raise ValueError("hop_ms must be positive and no larger than frame_ms")
    if min_f0_hz <= 0 or max_f0_hz <= min_f0_hz or audio.sample_rate <= 2 * max_f0_hz:
        raise ValueError("F0 bounds must be positive, ordered, and below Nyquist")
    if activity_rms_threshold < 0:
        raise ValueError("activity_rms_threshold must be non-negative")

    samples = np.clip(np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
    frame_size = max(1, round(audio.sample_rate * frame_ms / 1000.0))
    hop_size = max(1, round(audio.sample_rate * hop_ms / 1000.0))
    frame_f0: list[float] = []
    frame_rms: list[float] = []
    frame_zcr: list[float] = []
    frame_hnr: list[float] = []

    for start in range(0, samples.size, hop_size):
        part = samples[start:start + frame_size]
        if part.size < frame_size:
            part = np.pad(part, (0, frame_size - part.size))
        rms = float(np.sqrt(np.mean(np.square(part, dtype=np.float64))))
        zcr = float(np.mean(np.signbit(part[1:]) != np.signbit(part[:-1]))) if part.size > 1 else 0.0
        f0, hnr = _estimate_f0_and_hnr(part, audio.sample_rate, min_f0_hz, max_f0_hz)
        frame_f0.append(f0)
        frame_rms.append(rms)
        frame_zcr.append(zcr)
        frame_hnr.append(hnr)

    f0_values = np.asarray(frame_f0, dtype=np.float64)
    rms_values = np.asarray(frame_rms, dtype=np.float64)
    zcr_values = np.asarray(frame_zcr, dtype=np.float64)
    hnr_values = np.asarray(frame_hnr, dtype=np.float64)
    voiced = f0_values > 0.0
    voiced_f0 = f0_values[voiced]
    voiced_hnr = hnr_values[voiced]
    active = rms_values > activity_rms_threshold
    voiced_fraction = float(np.mean(voiced)) if voiced.size else 0.0
    activity_ratio = float(np.mean(active)) if active.size else 0.0
    pauses = 1.0 - activity_ratio

    f0_delta = _adjacent_deltas(f0_values, voiced, audio.sample_rate, hop_size)
    rms_delta = np.diff(rms_values) * audio.sample_rate / hop_size if rms_values.size > 1 else np.zeros(0)
    frame_times = np.arange(rms_values.size, dtype=np.float64) * hop_size / audio.sample_rate
    f0_times = frame_times[voiced]
    values = np.zeros(len(PROSODY_FEATURE_NAMES), dtype=np.float64)
    values[:13] = _pitch_summary(voiced_f0, f0_delta, f0_times)
    values[13:24] = _energy_summary(rms_values, rms_delta, frame_times)
    values[24] = voiced_fraction
    values[25] = pauses
    values[26] = activity_ratio
    values[27] = _longest_run_ratio(voiced)
    values[28] = float(np.mean(zcr_values)) if zcr_values.size else 0.0
    values[29] = float(np.std(zcr_values)) if zcr_values.size else 0.0
    values[30] = float(np.quantile(zcr_values, 0.9)) if zcr_values.size else 0.0
    values[31] = float(np.median(voiced_hnr)) if voiced_hnr.size else 0.0
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    return ProsodyFeatures(
        values=values,
        feature_names=PROSODY_FEATURE_NAMES,
        schema_version=PROSODY_SCHEMA_VERSION,
        sample_rate=audio.sample_rate,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        frame_length_samples=frame_size,
        hop_length_samples=hop_size,
        frame_count=rms_values.size,
        voiced_fraction=voiced_fraction,
    )


def _estimate_f0_and_hnr(frame: np.ndarray, sample_rate: int, min_f0_hz: float, max_f0_hz: float) -> tuple[float, float]:
    if frame.size < 4 or float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)))) < 1e-4:
        return 0.0, 0.0
    centered = frame.astype(np.float64) - float(np.mean(frame, dtype=np.float64))
    windowed = centered * np.hanning(frame.size)
    fft_size = 1 << (2 * frame.size - 1).bit_length()
    spectrum = np.fft.rfft(windowed, n=fft_size)
    autocorrelation = np.fft.irfft(spectrum * np.conjugate(spectrum), n=fft_size)[:frame.size]
    if autocorrelation[0] <= 0.0:
        return 0.0, 0.0
    normalized = autocorrelation / autocorrelation[0]
    lag_min = max(1, int(sample_rate / max_f0_hz))
    lag_max = min(frame.size - 2, int(sample_rate / min_f0_hz))
    if lag_max <= lag_min:
        return 0.0, 0.0
    segment = normalized[lag_min:lag_max + 1]
    peak_index = int(np.argmax(segment))
    lag = lag_min + peak_index
    periodicity = float(normalized[lag])
    if periodicity < 0.35:
        return 0.0, 0.0
    offset = 0.0
    if 0 < lag < frame.size - 1:
        left, center, right = normalized[lag - 1:lag + 2]
        denominator = left - 2.0 * center + right
        if abs(denominator) > 1e-12:
            offset = float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))
    f0 = sample_rate / (lag + offset)
    if not min_f0_hz <= f0 <= max_f0_hz:
        return 0.0, 0.0
    hnr = 10.0 * np.log10(max(periodicity, 1e-8) / max(1.0 - periodicity, 1e-8))
    return float(f0), float(np.clip(hnr, -20.0, 40.0))


def _adjacent_deltas(values: np.ndarray, valid: np.ndarray, sample_rate: int, hop_size: int) -> np.ndarray:
    if values.size < 2:
        return np.zeros(0)
    pairs = valid[1:] & valid[:-1]
    return np.diff(values)[pairs] * sample_rate / hop_size


def _pitch_summary(f0: np.ndarray, deltas: np.ndarray, times: np.ndarray) -> np.ndarray:
    result = np.zeros(13, dtype=np.float64)
    if f0.size:
        result[:10] = (
            np.median(f0), np.mean(f0), np.std(f0), np.min(f0),
            np.quantile(f0, 0.1), np.quantile(f0, 0.25), np.quantile(f0, 0.75),
            np.quantile(f0, 0.9), np.max(f0), np.max(f0) - np.min(f0),
        )
        result[12] = _slope(times, f0)
    if deltas.size:
        result[10:12] = (np.mean(deltas), np.std(deltas))
    return result


def _energy_summary(rms: np.ndarray, deltas: np.ndarray, times: np.ndarray) -> np.ndarray:
    result = np.zeros(11, dtype=np.float64)
    if rms.size:
        logged = np.log(np.maximum(rms, 1e-8))
        result[:9] = (
            np.mean(rms), np.std(rms), np.quantile(rms, 0.1), np.median(rms),
            np.quantile(rms, 0.9), np.max(rms) - np.min(rms),
            np.mean(logged), np.std(logged), _slope(times, rms),
        )
    if deltas.size:
        result[9:11] = (np.mean(deltas), np.std(deltas))
    return result


def _slope(times: np.ndarray, values: np.ndarray) -> float:
    if values.size < 2 or times.size != values.size or float(np.ptp(times)) <= 0.0:
        return 0.0
    return float(np.polyfit(times, values, 1)[0])


def _longest_run_ratio(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    longest = current = 0
    for value in mask:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest / mask.size
