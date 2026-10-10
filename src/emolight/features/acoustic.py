from dataclasses import dataclass
from enum import Enum

import numpy as np

from emolight.audio.vad import EnergyVAD
from emolight.audio.wav import AudioBuffer
from emolight.events import SystemStatus


class AudioQualityStatus(str, Enum):
    ACCEPTABLE = "ACCEPTABLE"
    LOW_QUALITY = "LOW_QUALITY"
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True)
class AcousticFeatures:
    duration_s: float
    rms: float
    zero_crossing_rate: float
    clipping_fraction: float
    audio_quality: float
    activity_ratio: float = 0.0
    active_frame_count: int = 0
    longest_active_run: int = 0
    estimated_snr_db: float | None = None
    quality_status: AudioQualityStatus = AudioQualityStatus.UNCERTAIN
    vad_frame_ms: float = 25.0


def extract_features(
    audio: AudioBuffer,
    *,
    frame_ms: float = 25.0,
    activity_rms_threshold: float = 0.01,
    min_activity_ratio: float = 0.20,
    min_contiguous_active_frames: int = 3,
    clipping_limit: float = 0.05,
    minimum_snr_db: float = 5.0,
    acceptable_snr_db: float = 10.0,
) -> AcousticFeatures:
    samples = np.asarray(audio.samples, dtype=np.float32)
    if samples.ndim != 1 or audio.sample_rate <= 0:
        raise ValueError("audio must be mono with a positive sample rate")
    if frame_ms <= 0 or activity_rms_threshold < 0:
        raise ValueError("frame_ms must be positive and activity threshold non-negative")
    if not 0.0 <= min_activity_ratio <= 1.0 or min_contiguous_active_frames < 1:
        raise ValueError("activity ratio must be bounded and contiguous frame count positive")
    if not 0.0 <= clipping_limit <= 1.0 or minimum_snr_db >= acceptable_snr_db:
        raise ValueError("clipping limit or SNR thresholds are invalid")
    if samples.size == 0:
        return AcousticFeatures(0.0, 0.0, 0.0, 0.0, 0.0, quality_status=AudioQualityStatus.LOW_QUALITY, vad_frame_ms=frame_ms)
    samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)
    rms = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))
    zcr = float(np.mean(np.signbit(samples[1:]) != np.signbit(samples[:-1]))) if len(samples) > 1 else 0.0
    clipping = float(np.mean(np.abs(samples) >= 0.999))
    frames = EnergyVAD(frame_ms=round(frame_ms), threshold_rms=activity_rms_threshold).process(
        AudioBuffer(samples, audio.sample_rate)
    )
    active = [frame.status is SystemStatus.UNCERTAIN for frame in frames]
    active_levels = np.asarray([frame.energy_rms for frame, is_active in zip(frames, active) if is_active])
    inactive_levels = np.asarray([frame.energy_rms for frame, is_active in zip(frames, active) if not is_active])
    activity_ratio = float(np.mean(active)) if active else 0.0
    active_frame_count = int(sum(active))
    longest_active_run = _longest_true_run(active)
    estimated_snr_db = None
    quality = 0.0
    if active_levels.size and inactive_levels.size and float(np.median(inactive_levels)) > 1e-6:
        noise_power = float(np.median(np.square(inactive_levels)))
        active_power = float(np.median(np.square(active_levels)))
        foreground_power = max(active_power - noise_power, 1e-12)
        estimated_snr_db = float(10.0 * np.log10(foreground_power / noise_power))
        quality = max(0.0, min(1.0, estimated_snr_db / 30.0))

    if clipping >= clipping_limit or activity_ratio < min_activity_ratio or longest_active_run < min_contiguous_active_frames:
        quality_status = AudioQualityStatus.LOW_QUALITY
    elif estimated_snr_db is None:
        quality_status = AudioQualityStatus.UNCERTAIN
    elif estimated_snr_db < minimum_snr_db:
        quality_status = AudioQualityStatus.LOW_QUALITY
    elif estimated_snr_db < acceptable_snr_db:
        quality_status = AudioQualityStatus.UNCERTAIN
    else:
        quality_status = AudioQualityStatus.ACCEPTABLE
    return AcousticFeatures(
        duration_s=len(samples) / audio.sample_rate,
        rms=rms,
        zero_crossing_rate=zcr,
        clipping_fraction=clipping,
        audio_quality=quality,
        activity_ratio=activity_ratio,
        active_frame_count=active_frame_count,
        longest_active_run=longest_active_run,
        estimated_snr_db=estimated_snr_db,
        quality_status=quality_status,
        vad_frame_ms=frame_ms,
    )


def _longest_true_run(values: list[bool]) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest
