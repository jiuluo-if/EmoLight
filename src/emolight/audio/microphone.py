from collections.abc import Callable
from dataclasses import dataclass
import queue
import threading
import time

import numpy as np


@dataclass(frozen=True)
class CapturedAudioFrame:
    samples: np.ndarray
    timestamp_ms: int


class MicrophoneAudioSource:
    """Optional sounddevice adapter; its device callback only enqueues bounded frames."""

    def __init__(self, sample_rate: int = 16000, frame_ms: int = 20, queue_frames: int = 16) -> None:
        if sample_rate <= 0 or frame_ms <= 0 or queue_frames <= 0:
            raise ValueError("sample_rate, frame_ms, and queue_frames must be positive")
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.blocksize = max(1, round(sample_rate * frame_ms / 1000))
        self._frames: queue.Queue[CapturedAudioFrame] = queue.Queue(maxsize=queue_frames)
        self._stream = None
        self._worker: threading.Thread | None = None
        self._running = False
        self.dropped_frames = 0
        self.last_error: str | None = None

    @property
    def worker_alive(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def start(self, on_audio: Callable[[CapturedAudioFrame], None]) -> None:
        if self._running:
            return
        if self._worker is not None and self._worker.is_alive():
            raise RuntimeError("previous microphone worker is still shutting down")
        try:
            import sounddevice as sd
        except ImportError as error:
            raise RuntimeError("microphone support requires the optional 'sounddevice' dependency") from error

        self._running = True
        self._worker = threading.Thread(target=self._consume, args=(on_audio,), daemon=True, name="emolight-audio")
        self._worker.start()

        def enqueue(indata, frames, time_info, status) -> None:
            if status:
                self.last_error = "audio callback warning"
            captured = CapturedAudioFrame(
                samples=indata[:, 0].copy(),
                timestamp_ms=round(time.monotonic() * 1000),
            )
            try:
                self._frames.put_nowait(captured)
            except queue.Full:
                try:
                    self._frames.get_nowait()
                    self.dropped_frames += 1
                except queue.Empty:
                    pass
                try:
                    self._frames.put_nowait(captured)
                except queue.Full:
                    self.dropped_frames += 1

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.blocksize,
                callback=enqueue,
            )
            self._stream.start()
        except Exception as error:
            self._running = False
            if self._stream is not None:
                self._stream.close()
                self._stream = None
            self._worker.join(timeout=1.0)
            if self._worker.is_alive():
                self.last_error = "microphone worker did not stop after stream startup failed"
            raise RuntimeError(f"cannot start microphone ({type(error).__name__})") from error

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        try:
            if stream is not None:
                try:
                    stream.stop()
                finally:
                    stream.close()
        except Exception as error:
            self.last_error = type(error).__name__
        finally:
            self._running = False
            while True:
                try:
                    self._frames.get_nowait()
                    self.dropped_frames += 1
                except queue.Empty:
                    break
            if self._worker is not None:
                self._worker.join(timeout=2.0)
                if self._worker.is_alive():
                    self.last_error = "microphone worker did not stop before timeout"

    def _consume(self, on_audio: Callable[[CapturedAudioFrame], None]) -> None:
        while self._running or not self._frames.empty():
            try:
                frame = self._frames.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                on_audio(frame)
            except Exception as error:
                self.last_error = type(error).__name__
