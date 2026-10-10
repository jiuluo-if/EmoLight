import csv
import json
import wave

import numpy as np
import pytest

from emolight.audio.wav import load_wav
from emolight.emotion.linear import NumpyLinearEmotionPredictor
from emolight.features.emotion_v1 import extract_emotion_features
from emolight.training.train_linear import _training_imputation_values, train_linear_models


def make_speaker_exclusive_manifest(root):
    audio_dir = root / "audio"
    audio_dir.mkdir()
    rows = []
    frequencies = {"neutral": 180.0, "happy": 260.0, "angry": 330.0, "sad": 130.0}
    rng = np.random.default_rng(7)
    for speaker in range(20):
        for emotion, f0 in frequencies.items():
            name = f"speaker-{speaker}-{emotion}.wav"
            samples = rng.normal(0.0, 0.002, 16000).astype(np.float32)
            times = np.arange(16000) / 16000
            amplitude = 0.10 + 0.005 * speaker
            samples[4000:] += (amplitude * np.sin(2.0 * np.pi * f0 * times[4000:])).astype(np.float32)
            pcm = np.clip(samples * 32767, -32768, 32767).astype("<i2")
            with wave.open(str(audio_dir / name), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(16000)
                stream.writeframes(pcm.tobytes())
            rows.append({
                "path": f"audio/{name}",
                "emotion": emotion,
                "speaker_id": f"speaker-{speaker}",
                "recording_id": f"recording-{speaker}-{emotion}",
                "dataset_id": "synthetic-test-only",
            })
    manifest = root / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def test_training_exports_simple_and_full_numpy_json_models_from_one_split(tmp_path, capsys):
    manifest = make_speaker_exclusive_manifest(tmp_path)
    output_dir = tmp_path / "models"
    noise_path = tmp_path / "fan.wav"
    fan_noise = (np.random.default_rng(101).normal(0.0, 0.02, 16000) * 32767).astype("<i2")
    with wave.open(str(noise_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(fan_noise.tobytes())
    noise_manifest = tmp_path / "noise_manifest.csv"
    with noise_manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("noise_type", "path"))
        writer.writeheader()
        writer.writerow({"noise_type": "fan", "path": noise_path.name})

    report = train_linear_models(
        manifest,
        output_dir,
        seed=31,
        white_noise_augmentation_snr_db=(20.0,),
        background_noise_manifest_path=noise_manifest,
        background_noise_augmentation_snr_db=(10.0,),
    )

    simple = json.loads((output_dir / "simple.json").read_text(encoding="utf-8"))
    full = json.loads((output_dir / "full.json").read_text(encoding="utf-8"))
    assert report["real_performance_evaluated"] is False
    assert "test_macro_f1" not in report
    assert report["augmentation"]["white_noise_snr_db"] == [20.0]
    assert report["augmentation"]["augmented_rows_by_snr"]["20dB"] > 0
    assert report["augmentation"]["background_noise_sources_sha256"]["fan"][0]
    assert report["augmentation"]["augmented_rows_by_snr"]["fan@10dB"] > 0
    assert simple["feature_profile"] == "simple"
    assert full["feature_profile"] == "full"
    assert simple["dataset_split"] == full["dataset_split"]
    assert simple["validation"]["threshold_selection_method"].startswith("maximize F1")
    assert simple["calibration_status"] == full["calibration_status"] == "CALIBRATED"
    assert report["models"]["simple"]["parameter_count"] > 0
    assert report["models"]["full"]["model_size_bytes"] < 100_000
    assert report["models"]["simple"]["file"] == "simple.json"
    assert all("path" not in model for model in report["models"].values())

    features = extract_emotion_features(load_wav(tmp_path / "audio" / "speaker-0-happy.wav"))
    simple_predictor = NumpyLinearEmotionPredictor.load(output_dir / "simple.json")
    full_predictor = NumpyLinearEmotionPredictor.load(output_dir / "full.json")
    assert simple_predictor.predict(features).model_status.value == "READY"
    assert full_predictor.predict(features).model_status.value == "READY"
    assert simple_predictor.predict(features).calibration_status == "CALIBRATED"
    assert full_predictor.predict(features).calibration_status == "CALIBRATED"

    from emolight.cli import run

    config_path = tmp_path / "settings.json"
    config_path.write_text(json.dumps({"audio": {
        "sample_rate_hz": 16000,
        "frame_ms": 25.0,
        "hop_ms": 10.0,
        "activity_rms_threshold": 0.01,
    }}), encoding="utf-8")
    wav_path = tmp_path / "audio" / "speaker-0-happy.wav"
    assert run([
        "--no-gui", "--emotion-only", "--wav", str(wav_path),
        "--model", str(output_dir / "full.json"), "--config", str(config_path),
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["identity_status"] == "NOT_EVALUATED"
    assert payload["target_event"] is None
    assert payload["prediction"]["model_status"] == "READY"


def test_training_records_dataset_and_live_window_provenance(tmp_path):
    manifest = make_speaker_exclusive_manifest(tmp_path)
    provenance = tmp_path / "dataset_metadata.json"
    provenance.write_text(json.dumps({
        "dataset_id": "test-real-corpus",
        "license": "CC-BY-4.0",
        "split_speaker_ids": {"train": ["a"], "validation": ["b"], "test": ["c"]},
        "nested_split": {"speaker_ids_by_split": {"train": ["private-id"]}},
        "official_gold_class_counts": {"train": {"happy": 3}, "test": {"happy": 1}},
        "split_window_class_counts": {"train": {"happy": 3}, "test": {"happy": 1}},
        "window_s": 1.5,
        "stride_s": 0.5,
    }), encoding="utf-8")

    train_linear_models(
        manifest,
        tmp_path / "models",
        dataset_metadata_path=provenance,
        window_s=1.5,
        update_interval_s=0.5,
    )

    artifact = json.loads((tmp_path / "models" / "full.json").read_text(encoding="utf-8"))
    assert artifact["dataset_provenance"]["dataset_id"] == "test-real-corpus"
    assert artifact["dataset_provenance"]["license"] == "CC-BY-4.0"
    assert artifact["window_s"] == 1.5
    assert artifact["update_interval_s"] == 0.5
    assert "test" not in artifact["dataset_provenance"]["official_gold_class_counts"]
    assert "test" not in artifact["dataset_provenance"]["split_window_class_counts"]
    assert "split_speaker_ids" not in artifact["dataset_provenance"]
    assert artifact["dataset_provenance"]["nested_split"] == {}


def test_optional_happy_weighted_logistic_model_is_validation_calibrated(tmp_path):
    manifest = make_speaker_exclusive_manifest(tmp_path)

    train_linear_models(
        manifest,
        tmp_path / "models",
        classifier_type="LogisticRegression",
        happy_class_weight=2.0,
    )

    artifact = json.loads((tmp_path / "models" / "full.json").read_text(encoding="utf-8"))
    assert artifact["training_library"]["classifier"] == "LogisticRegression"
    assert artifact["training_library"]["class_weight"]["happy"] == 2.0
    assert artifact["validation"]["metrics"]["sample_count"] > 0
    assert artifact["validation"]["threshold_selected_on"] == "validation"
    assert NumpyLinearEmotionPredictor.load(tmp_path / "models" / "full.json").model_status.value == "READY"


def test_training_refuses_to_invent_zero_for_an_all_missing_pitch_feature():
    matrix = np.ones((4, 24), dtype=np.float64)
    matrix[:, 0] = np.nan

    with pytest.raises(ValueError, match="no finite observations"):
        _training_imputation_values(matrix, "full")
