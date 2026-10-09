import csv

from emolight.data.manifest import load_manifest
from emolight.data.split import split_by_speaker_and_recording


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
