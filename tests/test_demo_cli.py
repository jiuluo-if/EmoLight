import json
import wave

import numpy as np

from emolight.cli import main, run
from emolight.demo import make_demo_event
from emolight.events import Emotion, EventSource


def test_demo_events_are_marked_and_cli_json_keeps_truth_boundary(capsys):
    event = make_demo_event(Emotion.ANGRY)
    assert event.source is EventSource.SIMULATION

    assert run(["--demo", "--no-gui", "--emotion", "angry"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "SIMULATION"
    assert payload["event"]["emotion"] == "angry"
    assert payload["lighting"]["simulation"] is True


def test_default_headless_status_reports_model_not_configured(capsys):
    assert run(["--no-gui"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "LIVE_UNCONFIGURED"
    assert payload["event"]["status"] == "NOT_CONFIGURED"
    assert payload["event"]["emotion"] is None
    assert payload["lighting"] is None


def test_cli_reads_wav_and_reports_features_without_guessing_emotion(tmp_path, capsys):
    path = tmp_path / "voice.wav"
    samples = (np.sin(2 * np.pi * 180 * np.arange(8000) / 8000) * 8000).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(samples.tobytes())

    assert run(["--no-gui", "--wav", str(path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["event"]["status"] == "NOT_CONFIGURED"
    assert payload["acoustic_features"]["duration_s"] == 1.0
    assert payload["event"]["emotion"] is None


def test_cli_reports_invalid_wav_without_traceback(capsys, tmp_path):
    result = run(["--no-gui", "--wav", str(tmp_path / "absent.wav")])

    assert result == 2
    assert "Traceback" not in capsys.readouterr().err


def test_installed_console_entrypoint_runs_headless(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["emolight", "--no-gui"])

    assert main() == 0
    assert json.loads(capsys.readouterr().out)["event"]["status"] == "NOT_CONFIGURED"
