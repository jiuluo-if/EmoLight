import csv
import hashlib
import io
import wave
import zipfile

import numpy as np

from emolight.data.manifest import load_manifest
from emolight.training.prepare_emodb import parse_emodb_filename, prepare_emodb_archive


def _wav_bytes(frequency_hz: float, sample_rate: int = 16000) -> bytes:
    samples = (0.15 * np.sin(2 * np.pi * frequency_hz * np.arange(sample_rate * 2) / sample_rate))
    pcm = np.clip(samples * 32767, -32768, 32767).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm.tobytes())
    return output.getvalue()


def _make_archive(path, wrong_sample_rate=False):
    labels = {"neutral": "N", "happiness": "F", "anger": "W", "sadness": "T", "fear": "A"}
    train_speakers = ("03", "08", "09", "10", "11", "13")
    test_speakers = ("12", "14", "15", "16")
    files = []
    train = []
    test = []
    next_frequency = 120
    for split, speakers, rows in (("train", train_speakers, train), ("test", test_speakers, test)):
        for speaker in speakers:
            for text_id, (emotion, code) in enumerate(labels.items(), start=1):
                if emotion == "fear" and split == "test":
                    continue
                name = f"{speaker}a{text_id:02d}{code}a.wav"
                rel = f"wav/{name}"
                files.append((rel, speaker))
                rows.append((rel, emotion))
                next_frequency += 1
                wav_data = _wav_bytes(next_frequency, 8000 if wrong_sample_rate and len(files) == 1 else 16000)
                # The test archive contains PCM bytes and the audformat CSVs, as the official archive does.
                with zipfile.ZipFile(path, "a", compression=zipfile.ZIP_DEFLATED) as archive:
                    archive.writestr(f"emodb/{rel}", wav_data)

    file_csv = io.StringIO(newline="")
    writer = csv.writer(file_csv)
    writer.writerow(("file", "duration", "speaker", "transcription"))
    writer.writerows((rel, "0 days 00:00:02", int(speaker), "a01") for rel, speaker in files)
    for target_rows, path_in_zip in (
        (train, "emodb/db.emotion.categories.train.gold_standard.csv"),
        (test, "emodb/db.emotion.categories.test.gold_standard.csv"),
    ):
        table = io.StringIO(newline="")
        table_writer = csv.writer(table)
        table_writer.writerow(("file", "emotion", "emotion.confidence"))
        table_writer.writerows((rel, emotion, "1.0") for rel, emotion in target_rows)
        with zipfile.ZipFile(path, "a", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(path_in_zip, table.getvalue())
    with zipfile.ZipFile(path, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("emodb/db.files.csv", file_csv.getvalue())


def test_emodb_filename_label_codes_do_not_mislabel_fear_as_anger():
    assert parse_emodb_filename("03a01Na.wav") == ("03", "neutral")
    assert parse_emodb_filename("03a01Fa.wav") == ("03", "happy")
    assert parse_emodb_filename("03a01Wa.wav") == ("03", "angry")
    assert parse_emodb_filename("03a01Ta.wav") == ("03", "sad")
    assert parse_emodb_filename("03a01Aa.wav") == ("03", None)


def test_preparation_preserves_official_test_speakers_and_emits_fixed_windows(tmp_path, monkeypatch):
    archive = tmp_path / "emodb.zip"
    _make_archive(archive)
    expected_md5 = hashlib.md5(archive.read_bytes()).hexdigest()
    monkeypatch.setattr("emolight.training.prepare_emodb.EMODB_ARCHIVE_MD5", expected_md5)
    output = tmp_path / "prepared"

    report = prepare_emodb_archive(archive, output, seed=42, window_s=1.5, stride_s=0.5)

    manifest = load_manifest(output / "emodb_manifest.csv")
    speaker_sets = {
        subset: {record.speaker_id for record in manifest.records if record.preassigned_split == subset}
        for subset in ("train", "validation", "test")
    }
    assert speaker_sets["test"] == {"12", "14", "15", "16"}
    assert len(speaker_sets["validation"]) == 2
    assert speaker_sets["train"].isdisjoint(speaker_sets["validation"])
    assert speaker_sets["train"].isdisjoint(speaker_sets["test"])
    assert speaker_sets["validation"].isdisjoint(speaker_sets["test"])
    assert {record.emotion.value for record in manifest.records} == {"neutral", "happy", "angry", "sad"}
    assert report["ignored_emotion_counts"] == {"fear": 6}
    assert report["window_count"] == len(manifest.records) == 80
    assert report["manifest_file"] == "emodb_manifest.csv"
    assert report["metadata_file"] == "dataset_metadata.json"
    assert "manifest_path" not in report and "metadata_path" not in report
    assert report["speaker_counts_by_split"] == {key: len(value) for key, value in speaker_sets.items()}
    assert "speaker_ids_by_split" not in report
    assert all(record.recording_id.startswith(record.speaker_id) for record in manifest.records)
    assert all(record.augmentation_group_id for record in manifest.records)


def test_preparation_rejects_audio_that_does_not_match_official_format(tmp_path, monkeypatch):
    archive = tmp_path / "emodb-wrong-rate.zip"
    _make_archive(archive, wrong_sample_rate=True)
    monkeypatch.setattr("emolight.training.prepare_emodb.EMODB_ARCHIVE_MD5", hashlib.md5(archive.read_bytes()).hexdigest())

    try:
        prepare_emodb_archive(archive, tmp_path / "prepared")
    except ValueError as error:
        assert "16 kHz PCM16" in str(error)
    else:
        raise AssertionError("expected sample-rate validation to fail")
