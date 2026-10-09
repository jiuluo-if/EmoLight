from collections.abc import Callable
import queue
import threading

import numpy as np


class MicrophoneAudioSource:
    """Optional sounddevice adapter; its device callback only enqueues bounded frames."""

    def __init__(self, sample_rate: int = 16000, frame_ms: int = 20, queue_frames: int = 16) -> None:
        if sample_rate <= 0 or frame_ms <= 0 or queue_frames <= 0:
            raise ValueError("sample_rate, frame_ms, and queue_frames must be positive")
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.blocksize = max(1, round(sample_rate * frame_ms / 1000))
        self._frames: queue.Queue[np.ndarray] = queue.Queue(maxsize=queue_frames)
        self._stream = None
        self._worker: threading.Thread | None = None
        self._running = False
        self.dropped_frames = 0
        self.last_error: str | None = None

    def start(self, on_audio: Callable[[np.ndarray], None]) -> None:
        if self._running:
            return
        try:
            import sounddevice as sd
        except ImportError as error:
            raise RuntimeError("microphone support requires the optional 'sounddevice' dependency") from error

        self._running = True
        self._worker = threading.Thread(target=self._consume, args=(on_audio,), daemon=True, name="emolight-audio")
        self._worker.start()

        def enqueue(indata, frames, time_info, status) -> None:
            if status:
                self.last_error = str(status)
            try:
                self._frames.put_nowait(indata[:, 0].copy())
            except queue.Full:
                try:
                    self._frames.get_nowait()
                    self.dropped_frames += 1
                except queue.Empty:
                    pass
                try:
                    self._frames.put_nowait(indata[:, 0].copy())
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
            self._worker = None
            raise RuntimeError(f"cannot start microphone: {error}") from error

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        try:
            if stream is not None:
                try:
                    stream.stop()
                finally:
                    stream.close()
        except Exception as error:
            self.last_error = str(error)
        finally:
            self._running = False
            if self._worker is not None:
                self._worker.join(timeout=2.0)
                self._worker = None

    def _consume(self, on_audio: Callable[[np.ndarray], None]) -> None:
        while self._running or not self._frames.empty():
            try:
                frame = self._frames.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                on_audio(frame)
            except Exception as error:
                self.last_error = str(error)
