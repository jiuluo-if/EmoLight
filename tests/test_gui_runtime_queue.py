import numpy as np

from emolight.audio.wav import AudioBuffer
from emolight.events import EmotionEvent, SystemStatus
from emolight.features.acoustic import AcousticFeatures, AudioQualityStatus
from emolight.gui.app import EmotionSimulatorApp
from emolight.runtime.pipeline import RuntimeSnapshot
from emolight.runtime.result_queue import LatestOnlyQueue


class StubRuntime:
    sample_rate = 16000

    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.feed_calls = 0

    def feed_snapshot(self, samples, timestamp_ms):
        self.feed_calls += 1
        self.last_timestamp_ms = timestamp_ms
        return self.snapshot


class TkCallForbidden:
    def after(self, *args):
        raise AssertionError("audio worker must not call Tk")


class TkPollRecorder:
    def __init__(self):
        self.scheduled = []

    def after(self, delay_ms, callback):
        self.scheduled.append((delay_ms, callback))


def test_audio_worker_publishes_snapshot_without_calling_tk():
    features = AcousticFeatures(1.0, 0.1, 0.2, 0.0, 0.8, quality_status=AudioQualityStatus.ACCEPTABLE)
    snapshot = RuntimeSnapshot(EmotionEvent.unknown(SystemStatus.NOT_CONFIGURED, 5), features, 5)
    app = EmotionSimulatorApp.__new__(EmotionSimulatorApp)
    app.runtime = StubRuntime(snapshot)
    app.result_queue = LatestOnlyQueue()
    app.root = TkCallForbidden()
    app._closed = False
    app.mic_source = object()

    app._process_audio_frame(type("CapturedFrame", (), {"samples": np.ones(10, dtype=np.float32), "timestamp_ms": 7})())

    assert app.result_queue.take_latest() is snapshot
    assert app.runtime.feed_calls == 1


def test_audio_worker_drops_queued_callback_after_microphone_stops():
    app = EmotionSimulatorApp.__new__(EmotionSimulatorApp)
    app.runtime = StubRuntime(None)
    app.result_queue = LatestOnlyQueue()
    app._closed = False
    app.mic_source = None

    app._process_audio_frame(type("CapturedFrame", (), {"samples": np.ones(10), "timestamp_ms": 9})())

    assert app.runtime.feed_calls == 0
    assert app.result_queue.take_latest() is None


def test_audio_worker_drops_frame_from_previous_microphone_session():
    app = EmotionSimulatorApp.__new__(EmotionSimulatorApp)
    app.runtime = StubRuntime(None)
    app.result_queue = LatestOnlyQueue()
    app._closed = False
    app.mic_source = object()
    app._mic_session_id = 2

    app._process_audio_frame(
        type("CapturedFrame", (), {"samples": np.ones(10), "timestamp_ms": 9})(),
        session_id=1,
    )

    assert app.runtime.feed_calls == 0


def test_ui_polls_latest_snapshot_on_main_thread():
    features = AcousticFeatures(1.0, 0.1, 0.2, 0.0, 0.8, quality_status=AudioQualityStatus.ACCEPTABLE)
    snapshot = RuntimeSnapshot(EmotionEvent.unknown(SystemStatus.NOT_CONFIGURED, 5), features, 5)
    root = TkPollRecorder()
    app = EmotionSimulatorApp.__new__(EmotionSimulatorApp)
    app.root = root
    app.result_queue = LatestOnlyQueue()
    app.result_queue.publish(snapshot)
    app._closed = False
    app._last_rendered_timestamp_ms = -1
    rendered = []
    app._show_live_status = rendered.append

    app._poll_results()

    assert rendered == [snapshot]
    assert len(root.scheduled) == 1
    assert root.scheduled[0][0] == 50


def test_gui_drops_runtime_snapshots_older_than_last_rendered_timestamp():
    features = AcousticFeatures(1.0, 0.1, 0.2, 0.0, 0.8, quality_status=AudioQualityStatus.ACCEPTABLE)
    newer = RuntimeSnapshot(EmotionEvent.unknown(SystemStatus.UNCERTAIN, 20), features, 20)
    older = RuntimeSnapshot(EmotionEvent.unknown(SystemStatus.UNCERTAIN, 10), features, 10)
    app = EmotionSimulatorApp.__new__(EmotionSimulatorApp)
    app._closed = False
    app.root = TkPollRecorder()
    app.result_queue = LatestOnlyQueue()
    app._last_rendered_timestamp_ms = 20
    rendered = []
    app._show_live_status = rendered.append
    app.result_queue.publish(older)

    app._poll_results()

    assert rendered == []


def test_gui_status_explains_unavailable_speaker_adapter_and_emotion_model():
    class StatusLabel:
        text = ""

        def configure(self, *, text):
            self.text = text

    class Config:
        speaker_model_path = "configured-speaker-model"

    app = EmotionSimulatorApp.__new__(EmotionSimulatorApp)
    app.config = Config()
    app.runtime = StubRuntime(None)
    app.runtime.predictor = type("Predictor", (), {"model_status": type("Status", (), {"value": "INVALID"})()})()
    app.model_load_error = "failed to load"
    app.status = StatusLabel()

    app._update_recognition_status("未接入")

    assert "目标身份：NOT_CONFIGURED" in app.status.text
    assert "声纹适配器尚未接入" in app.status.text
    assert "情绪模型：INVALID（模型加载失败）" in app.status.text


def test_microphone_start_requires_explicit_local_privacy_consent(monkeypatch):
    import emolight.gui.app as gui

    app = EmotionSimulatorApp.__new__(EmotionSimulatorApp)
    app.mic_source = None
    app.root = object()
    app.config = type("Config", (), {"audio": type("Audio", (), {"sample_rate_hz": 16000, "capture_frame_ms": 20})()})()
    app.runtime = StubRuntime(None)
    app._update_recognition_status = lambda status: None
    created = []
    monkeypatch.setattr(gui.messagebox, "askokcancel", lambda *args, **kwargs: False)
    monkeypatch.setattr(gui, "MicrophoneAudioSource", lambda **kwargs: created.append(kwargs))

    app._toggle_microphone()

    assert app.mic_source is None
    assert created == []


def test_stopping_microphone_clears_the_runtime_audio_window():
    app = EmotionSimulatorApp.__new__(EmotionSimulatorApp)
    cleared = []

    class Source:
        def stop(self):
            pass

    app.mic_source = Source()
    app._mic_session_id = 4
    app.runtime = StubRuntime(None)
    app.runtime.clear_audio_cache = lambda: cleared.append(True)
    app._update_recognition_status = lambda status: None

    app._toggle_microphone()

    assert app.mic_source is None
    assert cleared == [True]
