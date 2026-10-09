from dataclasses import dataclass

import numpy as np

from emolight.audio.wav import AudioBuffer
from emolight.events import SystemStatus


@dataclass(frozen=True)
class ActivityFrame:
    start_ms: int
    duration_ms: float
    energy_rms: float
    status: SystemStatus


class EnergyVAD:
    """Simple energy gate for local experiments; it does not identify speakers."""

    def __init__(self, frame_ms: int = 25, threshold_rms: float = 0.01) -> None:
        if frame_ms <= 0 or threshold_rms < 0:
            raise ValueError("frame_ms must be positive and threshold_rms non-negative")
        self.frame_ms = frame_ms
        self.threshold_rms = threshold_rms

    def process(self, audio: AudioBuffer) -> list[ActivityFrame]:
        frame_size = max(1, round(audio.sample_rate * self.frame_ms / 1000))
        frames: list[ActivityFrame] = []
        for start in range(0, len(audio.samples), frame_size):
            samples = np.asarray(audio.samples[start:start + frame_size], dtype=np.float32)
            if samples.size == 0:
                continue
            rms = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))
            status = SystemStatus.UNCERTAIN if rms >= self.threshold_rms else SystemStatus.SILENCE
            frames.append(ActivityFrame(
                start_ms=round(start * 1000 / audio.sample_rate),
                duration_ms=len(samples) * 1000 / audio.sample_rate,
                energy_rms=rms,
                status=status,
            ))
        return frames
