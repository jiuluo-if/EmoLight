import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Mapping

from emolight.events import Emotion


class ManifestError(ValueError):
    pass


@dataclass(frozen=True)
class ManifestRecord:
    path: Path
    emotion: Emotion
    speaker_id: str
    recording_id: str
    dataset_id: str
    file_sha256: str = ""
    source_recording_id: str = ""
    augmentation_group_id: str = ""


@dataclass(frozen=True)
class ManifestLoadResult:
    records: tuple[ManifestRecord, ...]
    skipped_labels: dict[str, int]
    label_mapping: dict[str, str]


_DEFAULT_LABEL_MAP = {
    "neutral": Emotion.NEUTRAL,
    "neu": Emotion.NEUTRAL,
    "n": Emotion.NEUTRAL,
    "happy": Emotion.HAPPY,
    "hap": Emotion.HAPPY,
    "f": Emotion.HAPPY,
    "angry": Emotion.ANGRY,
    "ang": Emotion.ANGRY,
    "w": Emotion.ANGRY,
    "sad": Emotion.SAD,
    "s": Emotion.SAD,
    "t": Emotion.SAD,
}


def load_manifest(
    path: str | Path,
    *,
    label_map: Mapping[str, str | Emotion] | None = None,
    base_dir: str | Path | None = None,
    skip_unmapped: bool = True,
) -> ManifestLoadResult:
    manifest_path = Path(path).resolve()
    root = Path(base_dir).resolve() if base_dir is not None else manifest_path.parent
    labels: dict[str, Emotion] = dict(_DEFAULT_LABEL_MAP)
    if label_map:
        for source, target in label_map.items():
            key = str(source).strip().casefold()
            try:
                labels[key] = target if isinstance(target, Emotion) else Emotion(str(target).strip().casefold())
            except ValueError as error:
                raise ManifestError(f"label map target {target!r} is not a supported emotion") from error

    records: list[ManifestRecord] = []
    skipped: dict[str, int] = {}
    content_labels: dict[str, Emotion] = {}
    required = {"path", "emotion", "speaker_id", "recording_id"}
    try:
        stream = manifest_path.open("r", encoding="utf-8-sig", newline="")
    except OSError as error:
        raise ManifestError(f"cannot open manifest: {error}") from error
    with stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or ()))
            raise ManifestError(f"manifest is missing required columns: {', '.join(missing)}")
        for row_number, row in enumerate(reader, start=2):
            raw_path = (row.get("path") or "").strip()
            raw_label = (row.get("emotion") or "").strip()
            speaker_id = (row.get("speaker_id") or "").strip()
            recording_id = (row.get("recording_id") or "").strip()
            dataset_id = (row.get("dataset_id") or "custom").strip() or "custom"
            source_recording_id = (row.get("source_recording_id") or "").strip() or recording_id
            augmentation_group_id = (row.get("augmentation_group_id") or "").strip()
            if not raw_path:
                raise ManifestError(f"row {row_number}: path is empty")
            if not speaker_id or not recording_id:
                raise ManifestError(f"row {row_number}: speaker_id and recording_id are required")
            if not raw_label:
                raise ManifestError(f"row {row_number}: emotion is empty")
            emotion = labels.get(raw_label.casefold())
            if emotion is None:
                if skip_unmapped:
                    skipped[raw_label] = skipped.get(raw_label, 0) + 1
                    continue
                raise ManifestError(f"row {row_number}: unmapped emotion label {raw_label!r}")
            audio_path = Path(raw_path)
            if not audio_path.is_absolute():
                audio_path = root / audio_path
            audio_path = audio_path.resolve()
            if not audio_path.is_file():
                raise ManifestError(f"row {row_number}: audio file does not exist: {raw_path}")
            digest = _sha256_file(audio_path)
            if digest:
                previous_emotion = content_labels.get(digest)
                if previous_emotion is not None and previous_emotion is not emotion:
                    raise ManifestError(
                        f"row {row_number}: same audio content has conflicting emotion labels "
                        f"{previous_emotion.value!r} and {emotion.value!r}"
                    )
                content_labels[digest] = emotion
            records.append(ManifestRecord(
                path=audio_path,
                emotion=emotion,
                speaker_id=speaker_id,
                recording_id=recording_id,
                dataset_id=dataset_id,
                file_sha256=digest,
                source_recording_id=source_recording_id,
                augmentation_group_id=augmentation_group_id,
            ))
    return ManifestLoadResult(
        records=tuple(records),
        skipped_labels=skipped,
        label_mapping={key: emotion.value for key, emotion in labels.items()},
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if path.stat().st_size == 0:
        return ""
    return digest.hexdigest()
