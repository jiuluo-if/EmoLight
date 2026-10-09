import csv
from dataclasses import replace

from emolight.data.manifest import load_manifest
from emolight.data.split import _connected_groups, split_by_speaker_and_recording


def make_manifest(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    rows = []
    for speaker_index in range(12):
        label = "neutral" if speaker_index % 2 == 0 else "happy"
        for recording_index in range(2):
            name = f"s{speaker_index}-r{recording_index}.wav"
            (audio_dir / name).touch()
            recording_id = f"s{speaker_index}-r{recording_index}"
            if speaker_index in (0, 2) and recording_index == 0:
                recording_id = "shared-recording"
            rows.append({
                "path": f"audio/{name}",
                "emotion": label,
                "speaker_id": f"s{speaker_index}",
                "recording_id": recording_id,
                "dataset_id": "toy-local",
            })
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return load_manifest(manifest).records


def test_split_keeps_speakers_and_recordings_exclusive(tmp_path):
    split = split_by_speaker_and_recording(make_manifest(tmp_path), seed=19)

    sets = [split.train, split.validation, split.test]
    for left_index, left in enumerate(sets):
        left_speakers = {(record.dataset_id, record.speaker_id) for record in left}
        left_recordings = {(record.dataset_id, record.recording_id) for record in left}
        for right in sets[left_index + 1:]:
            assert left_speakers.isdisjoint({(r.dataset_id, r.speaker_id) for r in right})
            assert left_recordings.isdisjoint({(r.dataset_id, r.recording_id) for r in right})
    assert split.train and split.validation and split.test
    assert len(split.train_group_ids) == len(split.train)
    assert len(split.validation_group_ids) == len(split.validation)
    assert len(split.test_group_ids) == len(split.test)


def test_split_rejects_too_few_independent_groups(tmp_path):
    records = make_manifest(tmp_path)
    one_group = tuple(record for record in records if record.speaker_id == "s0")

    try:
        split_by_speaker_and_recording(one_group)
    except ValueError as error:
        assert "at least 5" in str(error)
    else:
        raise AssertionError("expected insufficient independent speaker groups to fail")


def test_same_wav_with_different_speaker_and_recording_ids_stays_in_one_group(tmp_path):
    records = make_manifest(tmp_path)
    original = replace(records[0], file_sha256="duplicate-content-hash")
    records = (original,) + records[1:]
    duplicate_path = replace(
        original,
        speaker_id="spoofed-speaker-id",
        recording_id="spoofed-recording-id",
    )

    group_ids = _connected_groups(records + (duplicate_path,))

    assert group_ids[0] == group_ids[-1]
    split = split_by_speaker_and_recording(records + (duplicate_path,), seed=19)
    path_sets = [
        {record.path.resolve().as_posix().casefold() for record in subset}
        for subset in (split.train, split.validation, split.test)
    ]
    assert path_sets[0].isdisjoint(path_sets[1])
    assert path_sets[0].isdisjoint(path_sets[2])
    assert path_sets[1].isdisjoint(path_sets[2])


def test_duplicate_audio_copy_and_augmentation_versions_share_split_group(tmp_path):
    records = make_manifest(tmp_path)
    original = replace(records[0], file_sha256="duplicate-content-hash")
    records = (original,) + records[1:]
    duplicate_copy = replace(
        original,
        path=(original.path.parent / "copy.wav").resolve(),
        speaker_id="other-dataset-speaker",
        recording_id="copied-id",
        dataset_id="other-dataset",
        file_sha256="duplicate-content-hash",
    )
    augmented = replace(
        records[1],
        path=(records[1].path.parent / "augmented.wav").resolve(),
        speaker_id="augmentation-speaker",
        recording_id="augmentation-recording",
        source_recording_id=records[1].recording_id,
        augmentation_group_id="aug-group-1",
    )
    augmentation_base = replace(
        records[2],
        source_recording_id=records[1].recording_id,
        augmentation_group_id="aug-group-1",
    )

    group_ids = _connected_groups(records + (duplicate_copy, augmented, augmentation_base))

    assert group_ids[0] == group_ids[len(records)]
    assert group_ids[1] == group_ids[len(records) + 1]
    assert group_ids[1] == group_ids[len(records) + 2]


def test_split_metadata_reports_each_subset_class_and_speaker_counts(tmp_path):
    records = make_manifest(tmp_path)
    split = split_by_speaker_and_recording(records, seed=19)

    metadata = split.to_metadata()

    for name, subset in (("train", split.train), ("validation", split.validation), ("test", split.test)):
        summary = metadata["subsets"][name]
        assert summary["sample_count"] == len(subset)
        assert summary["speaker_count"] == len({(r.dataset_id, r.speaker_id) for r in subset})
        assert summary["class_counts"] == {
            label: sum(record.emotion.value == label for record in subset)
            for label in ("neutral", "happy", "angry", "sad")
        }
