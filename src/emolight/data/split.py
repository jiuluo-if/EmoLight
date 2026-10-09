from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from emolight.data.manifest import ManifestRecord


@dataclass(frozen=True)
class DatasetSplit:
    train: tuple[ManifestRecord, ...]
    validation: tuple[ManifestRecord, ...]
    test: tuple[ManifestRecord, ...]
    seed: int
    method: str
    group_counts: tuple[int, int, int]
    train_group_ids: tuple[int, ...]
    validation_group_ids: tuple[int, ...]
    test_group_ids: tuple[int, ...]

    def to_metadata(self) -> dict:
        return {
            "method": self.method,
            "seed": self.seed,
            "record_counts": {
                "train": len(self.train),
                "validation": len(self.validation),
                "test": len(self.test),
            },
            "group_counts": {
                "train": self.group_counts[0],
                "validation": self.group_counts[1],
                "test": self.group_counts[2],
            },
        }


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[tuple[str, str, str], tuple[str, str, str]] = {}

    def find(self, item: tuple[str, str, str]) -> tuple[str, str, str]:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: tuple[str, str, str], right: tuple[str, str, str]) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            if left_root < right_root:
                self.parent[right_root] = left_root
            else:
                self.parent[left_root] = right_root


def _connected_groups(records: tuple[ManifestRecord, ...]) -> np.ndarray:
    union = _UnionFind()
    keys = []
    for record in records:
        speaker = ("speaker", record.dataset_id, record.speaker_id)
        recording = ("recording", record.dataset_id, record.recording_id)
        union.union(speaker, recording)
        keys.append(speaker)
    roots = [union.find(key) for key in keys]
    root_ids = {root: index for index, root in enumerate(sorted(set(roots)))}
    return np.asarray([root_ids[root] for root in roots], dtype=np.int64)


def split_by_speaker_and_recording(records, *, seed: int = 42) -> DatasetSplit:
    records = tuple(records)
    if not records:
        raise ValueError("cannot split an empty manifest")
    groups = _connected_groups(records)
    group_count = int(np.unique(groups).size)
    if group_count < 5:
        raise ValueError("at least 5 independent speaker/recording groups are required")
    labels = np.asarray([record.emotion.value for record in records])
    if np.unique(labels).size < 2:
        raise ValueError("at least two emotion classes are required for a classifier split")

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = list(splitter.split(np.zeros(len(records)), labels, groups))
    validation_indices = set(map(int, folds[0][1]))
    test_indices = set(map(int, folds[1][1]))
    train_indices = set(range(len(records))) - validation_indices - test_indices
    split_indices = (train_indices, validation_indices, test_indices)
    split_records = [tuple(records[index] for index in sorted(indices)) for indices in split_indices]
    if len({record.emotion for record in split_records[0]}) < 2:
        raise ValueError("training split contains fewer than two emotion classes")
    split_groups = [set(groups[list(indices)].tolist()) for indices in split_indices]
    if any(split_groups[left] & split_groups[right] for left in range(3) for right in range(left + 1, 3)):
        raise RuntimeError("group splitter leaked a speaker or source recording across splits")
    return DatasetSplit(
        train=split_records[0],
        validation=split_records[1],
        test=split_records[2],
        seed=seed,
        method="StratifiedGroupKFold(5), speaker+recording connected groups; validation fold 0, test fold 1",
        group_counts=tuple(len(group_set) for group_set in split_groups),
        train_group_ids=tuple(int(groups[index]) for index in sorted(train_indices)),
        validation_group_ids=tuple(int(groups[index]) for index in sorted(validation_indices)),
        test_group_ids=tuple(int(groups[index]) for index in sorted(test_indices)),
    )
