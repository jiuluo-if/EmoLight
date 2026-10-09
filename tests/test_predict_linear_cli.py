import json
import wave

import numpy as np

from emolight.training.predict_linear import main


def test_predict_cli_missing_model_is_explicitly_not_configured(tmp_path, capsys):
    wav_path = tmp_path / "voice.wav"
    samples = (np.sin(2 * np.pi * 220 * np.arange(8000) / 8000) * 7000).astype(np.int16)
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(samples.tobytes())
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"audio": {"sample_rate_hz": 8000}}), encoding="utf-8")

    assert main(["--wav", str(wav_path), "--model", str(tmp_path / "missing.json"), "--config", str(config_path)]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "OFFLINE_EMOTION_ONLY"
    assert payload["identity_status"] == "NOT_EVALUATED"
    assert payload["target_event"] is None
    assert payload["prediction"]["model_status"] == "NOT_CONFIGURED"
    assert payload["prediction"]["emotion"] is None
