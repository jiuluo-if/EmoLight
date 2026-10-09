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


def extract_features(audio: AudioBuffer) -> AcousticFeatures:
    samples = np.asarray(audio.samples, dtype=np.float32)
    if samples.ndim != 1 or audio.sample_rate <= 0:
        raise ValueError("audio must be mono with a positive sample rate")
    if samples.size == 0:
        return AcousticFeatures(0.0, 0.0, 0.0, 0.0, 0.0, quality_status=AudioQualityStatus.LOW_QUALITY)
    samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)
    rms = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))
    zcr = float(np.mean(np.signbit(samples[1:]) != np.signbit(samples[:-1]))) if len(samples) > 1 else 0.0
    clipping = float(np.mean(np.abs(samples) >= 0.999))
    frames = EnergyVAD().process(AudioBuffer(samples, audio.sample_rate))
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

    if clipping >= 0.05 or activity_ratio < 0.20 or longest_active_run < 3:
        quality_status = AudioQualityStatus.LOW_QUALITY
    elif estimated_snr_db is None:
        quality_status = AudioQualityStatus.UNCERTAIN
    elif estimated_snr_db < 5.0:
        quality_status = AudioQualityStatus.LOW_QUALITY
    elif estimated_snr_db < 10.0:
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
    )


def _longest_true_run(values: list[bool]) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest
