import csv
import wave

import numpy as np

from emolight.emotion.predictor import ModelStatus
from emolight.emotion.sklearn_predictor import SklearnEmotionPredictor
from emolight.events import Emotion
from emolight.features.prosody import extract_prosody
from emolight.audio.wav import load_wav
from emolight.training.train_svm import train_from_manifest


def make_labeled_manifest(directory):
    audio_dir = directory / "audio"
    audio_dir.mkdir()
    rows = []
    for speaker in range(16):
        emotion = "neutral" if speaker % 2 == 0 else "happy"
        f0 = 210 if emotion == "neutral" else 310
        time = np.arange(16000) / 16000
        rng = np.random.default_rng(speaker)
        samples = rng.normal(0.0, 0.002, 16000)
        samples[4000:] += 0.15 * np.sin(2 * np.pi * f0 * time[4000:])
        samples = np.clip(samples * 32767, -32768, 32767).astype("<i2")
        filename = f"speaker-{speaker}.wav"
        with wave.open(str(audio_dir / filename), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(samples.tobytes())
        rows.append({
            "path": f"audio/{filename}",
            "emotion": emotion,
            "speaker_id": f"speaker-{speaker}",
            "recording_id": f"recording-{speaker}",
            "dataset_id": "synthetic-test-only",
        })
    manifest = directory / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def test_synthetic_manifest_can_exercise_training_and_emotion_only_inference(tmp_path):
    manifest = make_labeled_manifest(tmp_path)
    model_path = tmp_path / "local-model.joblib"

    result = train_from_manifest(manifest, model_path, seed=11)
    predictor = SklearnEmotionPredictor.load(model_path, expected_sample_rate=16000)
    features = extract_prosody(load_wav(tmp_path / "audio" / "speaker-0.wav"))
    prediction = predictor.predict(features)

    assert result["status"] == "TRAINED"
    assert result["real_performance_evaluated"] is False
    assert "macro_f1" not in result
    assert result["model_size_bytes"] == model_path.stat().st_size
    assert prediction.model_status is ModelStatus.READY
    assert prediction.emotion in set(Emotion)
