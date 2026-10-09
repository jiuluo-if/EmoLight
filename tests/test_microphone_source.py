import sys
import threading
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

    assert received[0].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert source.dropped_frames == 0


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
        received.append(int(frame[0]))
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
