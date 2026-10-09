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
