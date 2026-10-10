from dataclasses import dataclass

import numpy as np

from emolight.audio.wav import AudioBuffer


EMOTION_FEATURE_SCHEMA_VERSION = "emotion-prosody-24-v1"
EMOTION_FEATURE_NAMES = (
    "f0_median_hz", "f0_mean_hz", "f0_std_hz", "f0_range_hz",
    "f0_delta_mean_hz_s", "f0_delta_std_hz_s", "f0_slope_hz_s",
    "f0_valid_fraction", "mean_periodicity",
    "rms_mean", "rms_std", "rms_q10", "rms_q90", "rms_range",
    "log_rms_mean", "log_rms_std", "rms_delta_mean_s", "rms_delta_std_s", "rms_slope_s",
    "active_fraction", "pause_fraction", "longest_active_run_fraction",
    "zcr_mean", "hnr_median_db",
)
SIMPLE_FEATURE_SCHEMA_VERSION = "energy-rhythm-8-v1"
SIMPLE_FEATURE_NAMES = (
    "rms_mean", "rms_std", "rms_q10", "rms_q90",
    "log_rms_mean", "log_rms_std", "rms_delta_std_s", "pause_fraction",
)
F0_MISSING_INDICES = (0, 1, 2, 3, 4, 5, 6, 23)
_MIN_VALID_FRAME_COVERAGE = 0.8


@dataclass(frozen=True)
class EmotionFeatures:
    values: np.ndarray
    simple_values: np.ndarray
    feature_names: tuple[str, ...]
    schema_version: str
    simple_feature_names: tuple[str, ...]
    simple_schema_version: str
    sample_rate: int
    frame_ms: float
    hop_ms: float
    activity_rms_threshold: float
    valid_frame_fraction: float
    frame_count: int
    f0_valid_fraction: float
    mean_periodicity: float
    active_fraction: float

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=np.float32, copy=True)
        simple_values = np.array(self.simple_values, dtype=np.float32, copy=True)
        if values.shape != (24,) or self.feature_names != EMOTION_FEATURE_NAMES:
            raise ValueError("full emotion feature contract must contain the named 24 features")
        if simple_values.shape != (len(SIMPLE_FEATURE_NAMES),) or self.simple_feature_names != SIMPLE_FEATURE_NAMES:
            raise ValueError("simple emotion feature contract mismatch")
        if self.schema_version != EMOTION_FEATURE_SCHEMA_VERSION or self.simple_schema_version != SIMPLE_FEATURE_SCHEMA_VERSION:
            raise ValueError("emotion feature schema version mismatch")
        if not np.isfinite(simple_values).all():
            raise ValueError("simple emotion features must be finite")
        allowed_missing = np.zeros(values.shape, dtype=bool)
        allowed_missing[list(F0_MISSING_INDICES)] = True
        if np.isinf(values).any() or np.isnan(values[~allowed_missing]).any():
            raise ValueError("only unavailable F0-derived features may be NaN")
        if self.sample_rate <= 0 or self.frame_ms <= 0 or self.hop_ms <= 0 or self.hop_ms > self.frame_ms:
            raise ValueError("emotion feature timing metadata is invalid")
        for name in ("valid_frame_fraction", "f0_valid_fraction", "mean_periodicity", "active_fraction"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and between 0 and 1")
        if not np.isfinite(self.activity_rms_threshold) or self.activity_rms_threshold < 0.0 or self.frame_count < 0:
            raise ValueError("emotion feature thresholds or frame count are invalid")
        if self.f0_valid_fraction == 0.0 and not np.isnan(values[list(F0_MISSING_INDICES)]).all():
            raise ValueError("unavailable F0 summaries must remain explicitly missing")
        if self.f0_valid_fraction > 0.0 and np.isnan(values[list(F0_MISSING_INDICES)]).any():
            raise ValueError("valid F0 summaries cannot be marked missing")
        values.setflags(write=False)
        simple_values.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "simple_values", simple_values)


def extract_emotion_features(
    audio: AudioBuffer,
    *,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    activity_rms_threshold: float = 0.01,
    min_f0_hz: float = 60.0,
    max_f0_hz: float = 500.0,
    min_periodicity: float = 0.35,
) -> EmotionFeatures:
    """Extract causal energy/rhythm and prosody features with explicit frame validity."""
    original = np.asarray(audio.samples, dtype=np.float32)
    if original.ndim != 1 or audio.sample_rate <= 0:
        raise ValueError("audio must be mono with positive sample rate")
    if frame_ms <= 0 or hop_ms <= 0 or hop_ms > frame_ms:
        raise ValueError("frame/hop durations must be positive and hop must not exceed frame")
    if min_f0_hz <= 0 or max_f0_hz <= min_f0_hz or audio.sample_rate <= 2.0 * max_f0_hz:
        raise ValueError("F0 bounds must be ordered and below Nyquist")
    if (
        not np.isfinite(activity_rms_threshold)
        or activity_rms_threshold < 0.0
        or not np.isfinite(min_periodicity)
        or not 0.0 < min_periodicity <= 1.0
    ):
        raise ValueError("activity and periodicity thresholds are invalid")

    frame_size = max(1, round(audio.sample_rate * frame_ms / 1000.0))
    hop_size = max(1, round(audio.sample_rate * hop_ms / 1000.0))
    if original.size == 0:
        starts: range | tuple = ()
    else:
        starts = range(0, original.size, hop_size)

    rms_frames: list[float] = []
    zcr_frames: list[float] = []
    f0_frames: list[float] = []
    periodicity_frames: list[float] = []
    hnr_frames: list[float] = []
    valid_frames: list[bool] = []
    for start in starts:
        end = min(start + frame_size, original.size)
        raw_part = original[start:end]
        coverage = raw_part.size / frame_size
        frame_valid = coverage >= _MIN_VALID_FRAME_COVERAGE and bool(np.isfinite(raw_part).all())
        valid_frames.append(frame_valid)
        if not frame_valid:
            rms_frames.append(np.nan)
            zcr_frames.append(np.nan)
            f0_frames.append(np.nan)
            periodicity_frames.append(0.0)
            continue
        part = np.asarray(raw_part, dtype=np.float64)
        if part.size < frame_size:
            part = np.pad(part, (0, frame_size - part.size))
        part = np.clip(part, -1.0, 1.0)
        rms = float(np.sqrt(np.mean(np.square(part))))
        zcr = float(np.mean(np.signbit(part[1:]) != np.signbit(part[:-1]))) if part.size > 1 else 0.0
        f0, periodicity = _estimate_f0(part, audio.sample_rate, min_f0_hz, max_f0_hz)
        rms_frames.append(rms)
        zcr_frames.append(zcr)
        f0_frames.append(f0 if periodicity >= min_periodicity else np.nan)
        periodicity_frames.append(periodicity)
        if periodicity >= min_periodicity:
            hnr_frames.append(_periodicity_to_hnr(periodicity))

    valid_mask = np.asarray(valid_frames, dtype=bool)
    rms_all = np.asarray(rms_frames, dtype=np.float64)
    zcr_all = np.asarray(zcr_frames, dtype=np.float64)
    f0 = np.asarray(f0_frames, dtype=np.float64)
    periodicity = np.asarray(periodicity_frames, dtype=np.float64)
    rms = rms_all[valid_mask]
    zcr = zcr_all[valid_mask]
    voiced = np.isfinite(f0) & valid_mask
    voiced_f0 = f0[voiced]
    active_all = valid_mask & (rms_all > activity_rms_threshold)
    active = active_all[valid_mask]
    active_fraction = float(np.mean(active)) if active.size else 0.0
    f0_valid_fraction = float(np.sum(voiced) / max(1, int(np.sum(valid_mask))))
    valid_frame_fraction = float(np.mean(valid_frames)) if valid_frames else 0.0
    all_frame_times = np.arange(rms_all.size, dtype=np.float64) * hop_size / audio.sample_rate
    frame_times = all_frame_times[valid_mask]
    f0_delta = _adjacent_valid_deltas(f0, voiced, audio.sample_rate, hop_size)
    rms_pairs = valid_mask[1:] & valid_mask[:-1] if valid_mask.size > 1 else np.zeros(0, dtype=bool)
    rms_delta = np.diff(rms_all)[rms_pairs] * audio.sample_rate / hop_size if rms_pairs.size else np.zeros(0)

    values = np.zeros(24, dtype=np.float64)
    if voiced_f0.size:
        values[:4] = (
            np.median(voiced_f0), np.mean(voiced_f0), np.std(voiced_f0), np.ptp(voiced_f0),
        )
        values[6] = _linear_slope(all_frame_times[voiced], voiced_f0)
        values[23] = float(np.median(hnr_frames)) if hnr_frames else np.nan
        if f0_delta.size:
            values[4:6] = np.mean(f0_delta), np.std(f0_delta)
    else:
        values[:7] = np.nan
        values[23] = np.nan
    values[7] = f0_valid_fraction
    valid_periodicity = periodicity[valid_mask]
    values[8] = float(np.mean(valid_periodicity)) if valid_periodicity.size else 0.0
    values[9:19] = _energy_summary(rms, rms_delta, frame_times)
    values[19] = active_fraction
    values[20] = 1.0 - active_fraction
    values[21] = _longest_run_fraction(active_all)
    values[22] = float(np.mean(zcr)) if zcr.size else 0.0
    simple_indices = (9, 10, 11, 12, 14, 15, 17, 20)
    simple_values = values[list(simple_indices)]
    simple_values = np.nan_to_num(simple_values, nan=0.0, posinf=0.0, neginf=0.0)
    return EmotionFeatures(
        values=values,
        simple_values=simple_values,
        feature_names=EMOTION_FEATURE_NAMES,
        schema_version=EMOTION_FEATURE_SCHEMA_VERSION,
        simple_feature_names=SIMPLE_FEATURE_NAMES,
        simple_schema_version=SIMPLE_FEATURE_SCHEMA_VERSION,
        sample_rate=audio.sample_rate,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        activity_rms_threshold=activity_rms_threshold,
        valid_frame_fraction=valid_frame_fraction,
        frame_count=len(valid_frames),
        f0_valid_fraction=f0_valid_fraction,
        mean_periodicity=float(values[8]),
        active_fraction=active_fraction,
    )


def _estimate_f0(frame: np.ndarray, sample_rate: int, min_f0_hz: float, max_f0_hz: float) -> tuple[float, float]:
    frame_rms = float(np.sqrt(np.mean(np.square(frame))))
    if frame.size < 4 or frame_rms < 1e-5:
        return 0.0, 0.0
    centered = frame - float(np.mean(frame))
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
    periodicity = float(np.clip(normalized[lag], 0.0, 1.0))
    if periodicity < 1e-12:
        return 0.0, 0.0
    offset = 0.0
    if 0 < lag < frame.size - 1:
        left, center, right = normalized[lag - 1:lag + 2]
        denominator = left - 2.0 * center + right
        if abs(denominator) > 1e-12:
            offset = float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))
    f0 = sample_rate / (lag + offset)
    if not min_f0_hz <= f0 <= max_f0_hz:
        return 0.0, periodicity
    return float(f0), periodicity


def _periodicity_to_hnr(periodicity: float) -> float:
    numerator = max(periodicity, 1e-8)
    denominator = max(1.0 - periodicity, 1e-8)
    return float(np.clip(10.0 * np.log10(numerator / denominator), -20.0, 40.0))


def _adjacent_valid_deltas(values: np.ndarray, valid: np.ndarray, sample_rate: int, hop_size: int) -> np.ndarray:
    if values.size < 2:
        return np.zeros(0)
    adjacent = valid[1:] & valid[:-1]
    return np.diff(values)[adjacent] * sample_rate / hop_size


def _energy_summary(rms: np.ndarray, deltas: np.ndarray, times: np.ndarray) -> np.ndarray:
    result = np.zeros(10, dtype=np.float64)
    if rms.size:
        logged = np.log(np.maximum(rms, 1e-8))
        result[:7] = (
            np.mean(rms), np.std(rms), np.quantile(rms, 0.1), np.quantile(rms, 0.9),
            np.ptp(rms), np.mean(logged), np.std(logged),
        )
        result[9] = _linear_slope(times, rms)
    if deltas.size:
        result[7:9] = np.mean(deltas), np.std(deltas)
    return result


def _linear_slope(times: np.ndarray, values: np.ndarray) -> float:
    if values.size < 2 or times.size != values.size or float(np.ptp(times)) <= 0.0:
        return 0.0
    return float(np.polyfit(times, values, 1)[0])


def _longest_run_fraction(mask: np.ndarray) -> float:
    if not mask.size:
        return 0.0
    longest = current = 0
    for value in mask:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest / mask.size
