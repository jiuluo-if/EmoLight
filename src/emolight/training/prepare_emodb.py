"""Verify and prepare the official Berlin EmoDB 1.3.0 archive for training."""

import argparse
import csv
from collections import Counter
from itertools import combinations
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Iterable
from urllib.error import URLError
from urllib.request import Request, urlopen
import wave
import zipfile

import numpy as np


EMODB_RECORD_URL = "https://zenodo.org/records/7447302"
EMODB_DOWNLOAD_URL = "https://zenodo.org/records/7447302/files/emodb.zip?download=1"
EMODB_ARCHIVE_MD5 = "9d21362dbc5676ef3ab4745d83ced0db"
EMODB_DATASET_ID = "emodb-1.3.0"
EMODB_LICENSE = "CC-BY-4.0"
EMODB_ATTRIBUTION = (
    "Burkhardt, F., Paeschke, A., Kienast, M., Sendlmeier, W. F., & Weiss, B. "
    "Berlin EmoDB 1.3.0, Zenodo record 7447302, https://doi.org/10.5281/zenodo.7447302"
)

_FILENAME = re.compile(r"^(?P<speaker>\d{2})(?P<text>[ab]\d{2})(?P<emotion>[A-Z])(?P<version>[a-z])\.wav$", re.IGNORECASE)
_CODE_TO_EMOTION = {"N": "neutral", "F": "happy", "W": "angry", "T": "sad"}
_SOURCE_LABEL_TO_CODE = {
    "neutral": "N", "happiness": "F", "anger": "W", "sadness": "T",
    "fear": "A", "disgust": "E", "boredom": "L",
}
_SPLITS = ("train", "validation", "test")


def parse_emodb_filename(filename: str) -> tuple[str, str | None]:
    """Return the zero-padded speaker ID and four-class label for a gold WAV name."""
    match = _FILENAME.fullmatch(Path(filename).name)
    if match is None:
        raise ValueError(f"invalid EmoDB filename: {filename!r}")
    speaker = match.group("speaker")
    code = match.group("emotion").upper()
    return speaker, _CODE_TO_EMOTION.get(code)


def prepare_emodb_archive(
    archive_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 42,
    window_s: float = 1.5,
    stride_s: float = 0.5,
) -> dict:
    """Verify archive/official metadata and create speaker-exclusive fixed windows."""
    archive_path = Path(archive_path).resolve()
    output = Path(output_dir).resolve()
    if not np.isfinite(window_s) or not np.isfinite(stride_s) or window_s <= 0 or stride_s <= 0:
        raise ValueError("window and stride must be finite and positive")
    if stride_s > window_s:
        raise ValueError("window stride must not exceed the window duration")
    digest = _file_md5(archive_path)
    if digest != EMODB_ARCHIVE_MD5:
        raise ValueError(f"EmoDB archive MD5 mismatch: expected {EMODB_ARCHIVE_MD5}, got {digest}")

    try:
        archive = zipfile.ZipFile(archive_path)
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError(f"cannot open EmoDB zip archive: {error}") from error
    with archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise ValueError(f"EmoDB zip CRC failure: {corrupt}")
        file_rows = _read_csv(archive, "emodb/db.files.csv")
        train_rows = _read_csv(archive, "emodb/db.emotion.categories.train.gold_standard.csv")
        test_rows = _read_csv(archive, "emodb/db.emotion.categories.test.gold_standard.csv")
        speakers_by_file = {row["file"]: row["speaker"] for row in file_rows}
        source_splits = {"train": train_rows, "test": test_rows}
        source_speakers: dict[str, set[str]] = {}
        source_labels: dict[str, Counter[str]] = {}
        four_class_rows: dict[str, list[tuple[str, str, str]]] = {"train": [], "test": []}
        ignored: Counter[str] = Counter()
        seen_paths: set[str] = set()

        for split, rows in source_splits.items():
            split_speakers: set[str] = set()
            label_counts: Counter[str] = Counter()
            for row in rows:
                relative = _safe_archive_audio_path(row.get("file", ""))
                if relative in seen_paths:
                    raise ValueError(f"EmoDB gold tables contain duplicate recording: {relative}")
                seen_paths.add(relative)
                label = (row.get("emotion") or "").strip().casefold()
                expected_code = _SOURCE_LABEL_TO_CODE.get(label)
                if expected_code is None:
                    raise ValueError(f"unknown official EmoDB emotion label: {label!r}")
                speaker_from_name, mapped_emotion = parse_emodb_filename(PurePosixPath(relative).name)
                speaker_from_table = speakers_by_file.get(relative)
                if speaker_from_table is None or f"{int(speaker_from_table):02d}" != speaker_from_name:
                    raise ValueError(f"speaker id mismatch between filename and official table: {relative}")
                if _FILENAME.fullmatch(PurePosixPath(relative).name).group("emotion").upper() != expected_code:
                    raise ValueError(f"emotion code conflicts with official gold label for {relative}")
                split_speakers.add(speaker_from_name)
                label_counts[label] += 1
                if mapped_emotion is None:
                    ignored[label] += 1
                    continue
                four_class_rows[split].append((relative, speaker_from_name, mapped_emotion))
            source_speakers[split] = split_speakers
            source_labels[split] = label_counts

        if not source_speakers["train"].isdisjoint(source_speakers["test"]):
            raise ValueError("official EmoDB training and test tables share a speaker")
        archive_audio_files = {
            name[len("emodb/"):]
            for name in archive.namelist()
            if name.startswith("emodb/wav/") and name.casefold().endswith(".wav")
        }
        if archive_audio_files != seen_paths:
            raise ValueError("official EmoDB WAV files and gold train/test tables do not match exactly")
        for relative in sorted(archive_audio_files):
            _read_emodb_pcm(archive, f"emodb/{relative}")
        validation_speakers = _select_validation_speakers(
            four_class_rows["train"], source_speakers["train"], seed=seed
        )
        subset_speakers = {
            "validation": set(validation_speakers),
            "train": source_speakers["train"] - set(validation_speakers),
            "test": source_speakers["test"],
        }
        if not all(subset_speakers[name] for name in _SPLITS):
            raise ValueError("official EmoDB archive does not provide speakers for every requested subset")
        if any(
            subset_speakers[left] & subset_speakers[right]
            for index, left in enumerate(_SPLITS)
            for right in _SPLITS[index + 1:]
        ):
            raise RuntimeError("EmoDB speaker assignment overlaps across subsets")

        output.mkdir(parents=True, exist_ok=True)
        windows_dir = output / "windows"
        windows_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output / "emodb_manifest.csv"
        manifest_rows: list[dict[str, str]] = []
        class_counts = {subset: Counter() for subset in _SPLITS}
        window_samples = round(16000 * window_s)
        stride_samples = round(16000 * stride_s)
        if window_samples <= 0 or stride_samples <= 0:
            raise ValueError("window and stride are too short for 16 kHz audio")

        for source_split in ("train", "test"):
            for relative, speaker, emotion in four_class_rows[source_split]:
                subset = "test" if source_split == "test" else (
                    "validation" if speaker in subset_speakers["validation"] else "train"
                )
                source_name = PurePosixPath(relative).name
                recording_id = Path(source_name).stem
                samples = _read_emodb_pcm(archive, f"emodb/{relative}")
                for window_index, window in enumerate(_fixed_windows(samples, window_samples, stride_samples)):
                    filename = f"emodb_{speaker}_{recording_id}_{window_index:03d}.wav"
                    target = windows_dir / filename
                    _write_pcm16_wav(target, window, 16000)
                    manifest_rows.append({
                        "path": f"windows/{filename}",
                        "emotion": emotion,
                        "speaker_id": speaker,
                        "recording_id": recording_id,
                        "dataset_id": EMODB_DATASET_ID,
                        "source_recording_id": recording_id,
                        "augmentation_group_id": recording_id,
                        "split": subset,
                    })
                    class_counts[subset][emotion] += 1

    fieldnames = (
        "path", "emotion", "speaker_id", "recording_id", "dataset_id",
        "source_recording_id", "augmentation_group_id", "split",
    )
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    metadata = {
        "dataset_id": EMODB_DATASET_ID,
        "dataset_name": "Berlin Database of Emotional Speech (EmoDB)",
        "dataset_version": "1.3.0",
        "source_record": EMODB_RECORD_URL,
        "doi": "10.5281/zenodo.7447302",
        "license": EMODB_LICENSE,
        "attribution": EMODB_ATTRIBUTION,
        "archive_md5": digest,
        "archive_bytes": archive_path.stat().st_size,
        "archive_internal_license_metadata": "CC0-1.0; Zenodo record metadata is treated conservatively as CC-BY-4.0",
        "language": "German",
        "speech_type": "acted emotional utterances",
        "sample_rate_hz": 16000,
        "channels": 1,
        "bit_depth": 16,
        "label_code_mapping": {"N": "neutral", "F": "happy", "W": "angry", "T": "sad", "A": "ignored fear"},
        "ignored_emotion_counts": dict(sorted(ignored.items())),
        "official_gold_utterance_counts": {name: len(rows) for name, rows in source_splits.items()},
        "official_gold_class_counts": {name: dict(sorted(counts.items())) for name, counts in source_labels.items()},
        "split_speaker_ids": {name: sorted(values) for name, values in subset_speakers.items()},
        "split_window_class_counts": {name: dict(sorted(counts.items())) for name, counts in class_counts.items()},
        "window_s": window_s,
        "stride_s": stride_s,
        "window_count": len(manifest_rows),
        "seed": seed,
        "split_protocol": "official 1.3.0 gold test speakers remain held out; validation speakers selected only from official train speakers",
        "test_set_used_for_tuning": False,
    }
    metadata_path = output / "dataset_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {
        "status": "PREPARED",
        "manifest_file": manifest_path.name,
        "metadata_file": metadata_path.name,
        "window_count": len(manifest_rows),
        "ignored_emotion_counts": dict(sorted(ignored.items())),
        "class_counts_by_split": metadata["split_window_class_counts"],
        "speaker_counts_by_split": {name: len(values) for name, values in metadata["split_speaker_ids"].items()},
        "archive_md5": digest,
    }


def download_emodb_archive(path: str | Path) -> Path:
    """Download the pinned official Zenodo file and verify its published MD5."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and _file_md5(path) == EMODB_ARCHIVE_MD5:
        return path
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".download", dir=path.parent)
    import os

    os.close(fd)
    temporary = Path(temporary_name)
    try:
        request = Request(EMODB_DOWNLOAD_URL, headers={"User-Agent": "EmoLight dataset preparation"})
        with urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        digest = _file_md5(temporary)
        if digest != EMODB_ARCHIVE_MD5:
            raise ValueError(f"downloaded EmoDB archive MD5 mismatch: expected {EMODB_ARCHIVE_MD5}, got {digest}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _read_csv(archive: zipfile.ZipFile, filename: str) -> list[dict[str, str]]:
    try:
        with archive.open(filename) as binary, io.TextIOWrapper(binary, encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                raise ValueError(f"empty EmoDB metadata table: {filename}")
            return list(reader)
    except KeyError as error:
        raise ValueError(f"official EmoDB metadata is missing {filename}") from error


def _safe_archive_audio_path(value: str) -> str:
    path = PurePosixPath((value or "").strip())
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "wav" or path.suffix.casefold() != ".wav":
        raise ValueError(f"unsafe or invalid EmoDB audio path: {value!r}")
    return path.as_posix()


def _select_validation_speakers(
    rows: list[tuple[str, str, str]],
    train_speakers: set[str],
    *,
    seed: int,
) -> tuple[str, ...]:
    class_names = {"neutral", "happy", "angry", "sad"}
    candidates = []
    for validation in combinations(sorted(train_speakers), 2):
        validation_speakers = set(validation)
        validation_labels = {emotion for _, speaker, emotion in rows if speaker in validation_speakers}
        remaining_labels = {emotion for _, speaker, emotion in rows if speaker not in validation_speakers}
        if class_names.issubset(validation_labels) and class_names.issubset(remaining_labels):
            candidates.append(validation)
    if not candidates:
        raise ValueError("official training speakers cannot form four-class train and validation subsets")
    rng = np.random.default_rng(seed)
    return candidates[int(rng.integers(0, len(candidates)))]


def _read_emodb_pcm(archive: zipfile.ZipFile, member: str) -> np.ndarray:
    try:
        raw = archive.read(member)
    except KeyError as error:
        raise ValueError(f"official EmoDB archive is missing audio: {member}") from error
    try:
        with wave.open(io.BytesIO(raw), "rb") as stream:
            if stream.getcomptype() != "NONE" or stream.getnchannels() != 1 or stream.getframerate() != 16000 or stream.getsampwidth() != 2:
                raise ValueError(f"EmoDB audio must be uncompressed mono 16 kHz PCM16: {member}")
            frames = stream.readframes(stream.getnframes())
            if len(frames) != stream.getnframes() * 2:
                raise ValueError(f"incomplete PCM data in {member}")
    except (wave.Error, EOFError) as error:
        raise ValueError(f"invalid WAV file in EmoDB archive: {member}: {error}") from error
    samples = np.frombuffer(frames, dtype="<i2")
    if not samples.size:
        raise ValueError(f"empty EmoDB audio: {member}")
    return samples


def _fixed_windows(samples: np.ndarray, window_samples: int, stride_samples: int) -> Iterable[np.ndarray]:
    if samples.size <= window_samples:
        padded = np.zeros(window_samples, dtype=np.int16)
        padded[window_samples - samples.size:] = samples
        yield padded
        return
    last_start = samples.size - window_samples
    starts = list(range(0, last_start + 1, stride_samples))
    if starts[-1] != last_start:
        starts.append(last_start)
    for start in starts:
        yield samples[start:start + window_samples]


def _write_pcm16_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(np.asarray(samples, dtype="<i2").tobytes())


def _file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="prepare_emodb.py", description="Verify and prepare Berlin EmoDB 1.3.0 for speaker-exclusive four-class training")
    parser.add_argument("--archive", help="verified local Zenodo emodb.zip; if omitted, --download is required")
    parser.add_argument("--download", action="store_true", help="download the pinned official Zenodo archive")
    parser.add_argument("--output-dir", default="data/private/emodb-1.3.0/prepared")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-s", type=float, default=1.5)
    parser.add_argument("--stride-s", type=float, default=0.5)
    args = parser.parse_args(argv)
    try:
        if args.archive and args.download:
            parser.error("choose either --archive or --download")
        if not args.archive and not args.download:
            parser.error("provide --archive or --download")
        archive_path = Path(args.archive) if args.archive else download_emodb_archive(Path(args.output_dir).parent / "emodb.zip")
        report = prepare_emodb_archive(archive_path, args.output_dir, seed=args.seed, window_s=args.window_s, stride_s=args.stride_s)
    except (OSError, ValueError, RuntimeError, URLError) as error:
        parser.exit(2, f"prepare_emodb.py: operation failed ({type(error).__name__})\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
