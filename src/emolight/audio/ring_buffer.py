import numpy as np


class AudioRingBuffer:
    """Fixed-capacity mono float buffer with chronological snapshots."""

    def __init__(self, capacity_samples: int) -> None:
        if capacity_samples <= 0:
            raise ValueError("capacity_samples must be positive")
        self._data = np.zeros(capacity_samples, dtype=np.float32)
        self._write_index = 0
        self._size = 0

    @property
    def capacity(self) -> int:
        return self._data.size

    @property
    def size(self) -> int:
        return self._size

    def append(self, samples: np.ndarray) -> None:
        values = np.asarray(samples, dtype=np.float32).reshape(-1)
        if values.size >= self.capacity:
            self._data[:] = values[-self.capacity:]
            self._write_index = 0
            self._size = self.capacity
            return
        first = min(values.size, self.capacity - self._write_index)
        self._data[self._write_index:self._write_index + first] = values[:first]
        remaining = values.size - first
        if remaining:
            self._data[:remaining] = values[first:]
        self._write_index = (self._write_index + values.size) % self.capacity
        self._size = min(self.capacity, self._size + values.size)

    def snapshot(self) -> np.ndarray:
        start = (self._write_index - self._size) % self.capacity
        end = start + self._size
        if end <= self.capacity:
            return self._data[start:end].copy()
        return np.concatenate((self._data[start:], self._data[:end % self.capacity]))

    def clear(self) -> None:
        """Overwrite retained samples and reset the logical window."""
        self._data.fill(0.0)
        self._write_index = 0
        self._size = 0
