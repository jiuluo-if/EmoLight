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

    def feed_snapshot(self, samples, timestamp_ms):
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

    app._process_audio_frame(np.ones(10, dtype=np.float32))

    assert app.result_queue.take_latest() is snapshot


def test_ui_polls_latest_snapshot_on_main_thread():
    features = AcousticFeatures(1.0, 0.1, 0.2, 0.0, 0.8, quality_status=AudioQualityStatus.ACCEPTABLE)
    snapshot = RuntimeSnapshot(EmotionEvent.unknown(SystemStatus.NOT_CONFIGURED, 5), features, 5)
    root = TkPollRecorder()
    app = EmotionSimulatorApp.__new__(EmotionSimulatorApp)
    app.root = root
    app.result_queue = LatestOnlyQueue()
    app.result_queue.publish(snapshot)
    app._closed = False
    rendered = []
    app._show_live_status = rendered.append

    app._poll_results()

    assert rendered == [snapshot]
    assert len(root.scheduled) == 1
    assert root.scheduled[0][0] == 50


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
