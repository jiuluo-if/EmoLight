from dataclasses import dataclass
from pathlib import Path
import wave

import numpy as np


@dataclass(frozen=True)
class AudioBuffer:
    samples: np.ndarray
    sample_rate: int

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.sample_rate


def load_wav(path: str | Path) -> AudioBuffer:
    """Read uncompressed PCM WAV and downmix channels to mono float32 [-1, 1]."""
    with wave.open(str(path), "rb") as wav:
        if wav.getcomptype() != "NONE":
            raise ValueError("only uncompressed PCM WAV is supported")
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if channels < 1 or rate < 1 or width not in (1, 2, 3, 4):
        raise ValueError("WAV must have a valid sample rate, channel count, and PCM width")

    if width == 1:
        raw = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        samples = (raw - 128.0) / 128.0
    elif width == 2:
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        bytes_ = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        values = bytes_[:, 0] | (bytes_[:, 1] << 8) | (bytes_[:, 2] << 16)
        values = np.where(values & 0x800000 != 0, values - 0x1000000, values)
        samples = values.astype(np.float32) / 8388608.0
    else:
        samples = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0

    if len(samples) % channels:
        raise ValueError("WAV frame data is incomplete")
    mono = samples.reshape(-1, channels).mean(axis=1) if channels > 1 else samples
    return AudioBuffer(np.ascontiguousarray(mono, dtype=np.float32), rate)
