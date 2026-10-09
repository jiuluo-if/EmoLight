import csv

import pytest

from emolight.data.manifest import ManifestError, load_manifest


def write_csv(path, rows):
    fields = (
        "path", "emotion", "speaker_id", "recording_id", "dataset_id",
        "source_recording_id", "augmentation_group_id",
    )
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


def test_manifest_records_normalized_path_hash_and_augmentation_lineage(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    source = audio_dir / "source.wav"
    duplicate = audio_dir / "copy.wav"
    source.write_bytes(b"same local test audio")
    duplicate.write_bytes(source.read_bytes())
    manifest_path = tmp_path / "manifest.csv"
    write_csv(manifest_path, [
        {"path": "audio/source.wav", "emotion": "happy", "speaker_id": "spk1", "recording_id": "r1", "dataset_id": "d1", "source_recording_id": "source-r1"},
        {"path": "audio/copy.wav", "emotion": "happy", "speaker_id": "spk2", "recording_id": "r2", "dataset_id": "d2", "augmentation_group_id": "aug-source-r1"},
    ])

    records = load_manifest(manifest_path).records

    assert records[0].path == source.resolve()
    assert records[0].file_sha256 == records[1].file_sha256
    assert records[0].source_recording_id == "source-r1"
    assert records[1].augmentation_group_id == "aug-source-r1"


def test_manifest_rejects_same_audio_content_with_conflicting_emotion_labels(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    first = audio_dir / "first.wav"
    second = audio_dir / "second.wav"
    first.write_bytes(b"same audio content")
    second.write_bytes(first.read_bytes())
    manifest_path = tmp_path / "conflicting.csv"
    write_csv(manifest_path, [
        {"path": "audio/first.wav", "emotion": "happy", "speaker_id": "s1", "recording_id": "r1", "dataset_id": "d1"},
        {"path": "audio/second.wav", "emotion": "sad", "speaker_id": "s2", "recording_id": "r2", "dataset_id": "d2"},
    ])

    with pytest.raises(ManifestError, match="same audio content.*conflicting emotion labels"):
        load_manifest(manifest_path)
