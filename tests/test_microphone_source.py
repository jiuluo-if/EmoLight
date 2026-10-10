import sys
import threading
import time
import types

import numpy as np

from emolight.audio.microphone import MicrophoneAudioSource


def test_microphone_callback_enqueues_frames_for_worker(monkeypatch):
    received = []
    ready = threading.Event()

    class FakeInputStream:
        def __init__(self, **kwargs):
            self.callback = kwargs["callback"]

        def start(self):
            self.callback(np.ones((4, 1), dtype=np.float32), 4, {}, None)

        def stop(self):
            pass

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(InputStream=FakeInputStream))

    source = MicrophoneAudioSource(sample_rate=16000, frame_ms=1)
    source.start(lambda frame: (received.append(frame), ready.set()))
    assert ready.wait(1.0)
    source.stop()

    assert received[0].samples.tolist() == [1.0, 1.0, 1.0, 1.0]
    assert received[0].timestamp_ms > 0
    assert source.dropped_frames == 0
    assert source._worker is not None and not source._worker.is_alive()


def test_microphone_overload_drops_oldest_queued_frame(monkeypatch):
    received = []
    first_frame_started = threading.Event()
    release_worker = threading.Event()
    all_frames_received = threading.Event()

    class FakeInputStream:
        def __init__(self, **kwargs):
            self.callback = kwargs["callback"]

        def start(self):
            for value in (1, 2, 3, 4):
                self.callback(np.full((4, 1), value, dtype=np.float32), 4, {}, None)
                if value == 1:
                    assert first_frame_started.wait(1.0)

        def stop(self):
            pass

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(InputStream=FakeInputStream))
    source = MicrophoneAudioSource(sample_rate=16000, frame_ms=1, queue_frames=2)

    def consume(frame):
        received.append(int(frame.samples[0]))
        if len(received) == 1:
            first_frame_started.set()
            release_worker.wait(1.0)
        if len(received) == 3:
            all_frames_received.set()

    source.start(consume)
    release_worker.set()
    assert all_frames_received.wait(1.0)
    source.stop()

    assert received == [1, 3, 4]
    assert source.dropped_frames == 1
    assert source._worker is not None and not source._worker.is_alive()


def test_stop_discards_pending_audio_and_joins_the_worker(monkeypatch):
    first_started = threading.Event()
    release_worker = threading.Event()
    received = []

    class FakeInputStream:
        def __init__(self, **kwargs):
            self.callback = kwargs["callback"]

        def start(self):
            self.callback(np.full((4, 1), 1, dtype=np.float32), 4, {}, None)
            assert first_started.wait(1.0)
            self.callback(np.full((4, 1), 2, dtype=np.float32), 4, {}, None)
            self.callback(np.full((4, 1), 3, dtype=np.float32), 4, {}, None)

        def stop(self):
            pass

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(InputStream=FakeInputStream))
    source = MicrophoneAudioSource(sample_rate=16000, frame_ms=1, queue_frames=4)

    def consume(frame):
        received.append(int(frame.samples[0]))
        first_started.set()
        release_worker.wait(1.0)

    source.start(consume)
    stop_thread = threading.Thread(target=source.stop)
    stop_thread.start()
    deadline = time.monotonic() + 0.5
    while not source._frames.empty() and time.monotonic() < deadline:
        time.sleep(0.001)
    assert source._frames.empty()
    release_worker.set()
    stop_thread.join(1.0)

    assert not stop_thread.is_alive()
    assert received == [1]
    assert source.dropped_frames == 2
    assert source._worker is not None and not source._worker.is_alive()
