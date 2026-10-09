import csv

import pytest

from emolight.data.manifest import ManifestError, load_manifest


def write_csv(path, rows):
    fields = ("path", "emotion", "speaker_id", "recording_id", "dataset_id")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_manifest_resolves_local_paths_maps_labels_and_skips_unsupported(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "sample.wav").touch()
    (audio_dir / "unsupported.wav").touch()
    manifest_path = tmp_path / "manifest.csv"
    write_csv(manifest_path, [
        {"path": "audio/sample.wav", "emotion": "HAP", "speaker_id": "spk1", "recording_id": "rec1", "dataset_id": "CREMA-D"},
        {"path": "audio/unsupported.wav", "emotion": "DIS", "speaker_id": "spk2", "recording_id": "rec2", "dataset_id": "CREMA-D"},
    ])

    result = load_manifest(manifest_path)

    assert len(result.records) == 1
    assert result.records[0].path == (audio_dir / "sample.wav").resolve()
    assert result.records[0].emotion.value == "happy"
    assert result.records[0].dataset_id == "CREMA-D"
    assert result.skipped_labels == {"DIS": 1}
    assert result.label_mapping["hap"] == "happy"


def test_manifest_requires_speaker_and_original_recording_ids(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    write_csv(manifest_path, [
        {"path": "missing.wav", "emotion": "neutral", "speaker_id": "", "recording_id": "", "dataset_id": "custom"},
    ])

    with pytest.raises(ManifestError, match="speaker_id and recording_id"):
        load_manifest(manifest_path)


def test_manifest_supports_user_label_mapping(tmp_path):
    (tmp_path / "sample.wav").touch()
    manifest_path = tmp_path / "manifest.csv"
    write_csv(manifest_path, [
        {"path": "sample.wav", "emotion": "Joy", "speaker_id": "spk1", "recording_id": "rec1", "dataset_id": "local"},
    ])

    result = load_manifest(manifest_path, label_map={"Joy": "happy"})

    assert result.records[0].emotion.value == "happy"
    assert result.label_mapping["joy"] == "happy"
