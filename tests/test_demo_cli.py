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


def test_wav_analysis_passes_audio_quality_settings_from_config(tmp_path, capsys, monkeypatch):
    import emolight.cli as cli

    wav_path = tmp_path / "configured.wav"
    samples = (np.sin(2 * np.pi * 180 * np.arange(8000) / 8000) * 5000).astype(np.int16)
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(samples.tobytes())
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"audio": {
        "frame_ms": 20.0,
        "activity_rms_threshold": 0.03,
        "min_activity_ratio": 0.35,
        "min_contiguous_active_frames": 4,
        "clipping_limit": 0.03,
        "minimum_snr_db": 6.0,
        "acceptable_snr_db": 12.0,
    }}), encoding="utf-8")
    real_extract_features = cli.extract_features
    observed = {}

    def capture_settings(audio, **kwargs):
        observed.update(kwargs)
        return real_extract_features(audio, **kwargs)

    monkeypatch.setattr(cli, "extract_features", capture_settings)
    assert run(["--no-gui", "--wav", str(wav_path), "--config", str(config_path)]) == 0
    capsys.readouterr()

    assert observed == {
        "frame_ms": 20.0,
        "activity_rms_threshold": 0.03,
        "min_activity_ratio": 0.35,
        "min_contiguous_active_frames": 4,
        "clipping_limit": 0.03,
        "minimum_snr_db": 6.0,
        "acceptable_snr_db": 12.0,
    }


def test_cli_reports_invalid_wav_without_traceback(capsys, tmp_path):
    result = run(["--no-gui", "--wav", str(tmp_path / "absent.wav")])

    assert result == 2
    assert "Traceback" not in capsys.readouterr().err


def test_installed_console_entrypoint_runs_headless(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["emolight", "--no-gui"])

    assert main() == 0
    assert json.loads(capsys.readouterr().out)["event"]["status"] == "NOT_CONFIGURED"


def test_offline_emotion_only_never_claims_target_identity_when_model_is_missing(tmp_path, capsys):
    wav_path = tmp_path / "offline.wav"
    samples = np.full(16000, 50, dtype=np.int16)
    times = np.arange(8000) / 16000
    samples[8000:] += (np.sin(2 * np.pi * 220 * times) * 3200).astype(np.int16)
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(samples.tobytes())

    assert run([
        "--no-gui", "--emotion-only", "--wav", str(wav_path),
        "--model", str(tmp_path / "missing.json"),
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "OFFLINE_EMOTION_ONLY"
    assert payload["identity_status"] == "NOT_EVALUATED"
    assert payload["target_event"] is None
    assert payload["prediction"]["status"] == "NOT_CONFIGURED"
    assert payload["prediction"]["emotion"] is None
