from dataclasses import dataclass

import numpy as np

from emolight.audio.wav import AudioBuffer


@dataclass(frozen=True)
class AcousticFeatures:
    duration_s: float
    rms: float
    zero_crossing_rate: float
    clipping_fraction: float
    audio_quality: float


def extract_features(audio: AudioBuffer) -> AcousticFeatures:
    samples = np.asarray(audio.samples, dtype=np.float32)
    if samples.ndim != 1 or audio.sample_rate <= 0:
        raise ValueError("audio must be mono with a positive sample rate")
    if samples.size == 0:
        return AcousticFeatures(0.0, 0.0, 0.0, 0.0, 0.0)
    samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)
    rms = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))
    zcr = float(np.mean(np.signbit(samples[1:]) != np.signbit(samples[:-1]))) if len(samples) > 1 else 0.0
    clipping = float(np.mean(np.abs(samples) >= 0.999))
    level_score = min(1.0, rms / 0.03)
    quality = max(0.0, min(1.0, level_score * (1.0 - clipping)))
    return AcousticFeatures(
        duration_s=len(samples) / audio.sample_rate,
        rms=rms,
        zero_crossing_rate=zcr,
        clipping_fraction=clipping,
        audio_quality=quality,
    )
