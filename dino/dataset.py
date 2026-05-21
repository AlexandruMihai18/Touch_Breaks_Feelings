from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_REGISTRY: dict[str, Path] = {
    "epic_kitchen": PROJECT_ROOT / "data" / "epic_kitchen" / "annotations",
    "greatest_hits": PROJECT_ROOT / "data" / "greatest_hits" / "annotations",
    "manual": PROJECT_ROOT / "data" / "manual_annotations" / "annotations",
    "kubric_movi_a_256_test": PROJECT_ROOT / "data" / "kubric_movi_a_256_test" / "annotations",
}


@dataclass(frozen=True)
class DinoSample:
    image_path: str
    label: int
    dataset: str
    video_id: str
    frame_id: str
    sample_type: str


@dataclass(frozen=True)
class SplitRows:
    train: list[DinoSample]
    test: list[DinoSample]


def normalize_row(row: dict, dataset_name: str) -> DinoSample:
    """Normalize any supported annotation row into an image-level classifier sample."""
    image_path = row.get("image_path")
    if not image_path:
        raise ValueError(f"Annotation row in {dataset_name!r} is missing image_path")

    sample_type = str(row.get("type", "no-touch"))
    frame_id = (
        row.get("frame_id")
        or row.get("frame")
        or (row.get("kubric") or {}).get("frame")
        or Path(str(image_path)).stem
    )
    return DinoSample(
        image_path=str(image_path),
        label=1 if sample_type == "touch" else 0,
        dataset=dataset_name,
        video_id=str(row.get("video_id", "")),
        frame_id=str(frame_id),
        sample_type=sample_type,
    )


def load_annotation_file(path: str | Path, dataset_name: str) -> list[DinoSample]:
    path = Path(path)
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list of annotation rows in {path}")
    return [normalize_row(row, dataset_name) for row in rows]


def _dataset_name_for_root(root: Path) -> str:
    parent = root.parent.name
    return parent if parent and parent != "data" else root.name


def resolve_annotation_roots(
    datasets: Iterable[str] | None,
    annotation_roots: Iterable[str | Path] | None,
) -> list[tuple[str, Path]]:
    roots: list[tuple[str, Path]] = []
    for name in datasets or []:
        if name not in DATASET_REGISTRY:
            raise ValueError(
                f"Unknown dataset {name!r}. Available: {', '.join(sorted(DATASET_REGISTRY))}"
            )
        roots.append((name, DATASET_REGISTRY[name]))

    for raw_root in annotation_roots or []:
        root = Path(raw_root).expanduser().resolve()
        roots.append((_dataset_name_for_root(root), root))

    if not roots:
        raise ValueError("Provide at least one --datasets value or --annotation-root path")
    return roots


def _split_key(sample: DinoSample) -> str:
    video = sample.video_id or Path(sample.image_path).parent.name
    return f"{video}::{sample.label}"


def split_samples(
    samples: list[DinoSample],
    test_size: float,
    seed: int,
) -> tuple[list[DinoSample], list[DinoSample]]:
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    if len(samples) < 2:
        raise ValueError("Need at least two samples to split")

    keys = [_split_key(s) for s in samples]
    counts = Counter(keys)
    stratify = keys if all(v >= 2 for v in counts.values()) and len(counts) > 1 else None

    train_rows, test_rows = train_test_split(
        samples,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    return list(train_rows), list(test_rows)


def _load_single_root(
    dataset_name: str,
    root: Path,
    auto_split: bool,
    auto_split_test_size: float,
    seed: int,
) -> SplitRows:
    if not root.exists():
        raise FileNotFoundError(f"Annotation root does not exist: {root}")

    train_path = root / "train.json"
    test_path = root / "test.json"
    val_path = root / "val.json"

    if train_path.exists():
        train_rows = load_annotation_file(train_path, dataset_name)
        if test_path.exists():
            test_rows = load_annotation_file(test_path, dataset_name)
        elif val_path.exists():
            test_rows = load_annotation_file(val_path, dataset_name)
        elif auto_split:
            train_rows, test_rows = split_samples(train_rows, auto_split_test_size, seed)
        else:
            raise FileNotFoundError(
                f"{root} has train.json but no test.json or val.json. "
                "Pass --auto-split to split train.json in memory."
            )
        return SplitRows(train=train_rows, test=test_rows)

    split_candidates = [p for p in (test_path, val_path) if p.exists()]
    if len(split_candidates) == 1 and auto_split:
        rows = load_annotation_file(split_candidates[0], dataset_name)
        train_rows, test_rows = split_samples(rows, auto_split_test_size, seed)
        return SplitRows(train=train_rows, test=test_rows)

    if len(split_candidates) == 1:
        raise FileNotFoundError(
            f"{root} only has {split_candidates[0].name}. Pass --auto-split to create "
            "train/test splits in memory."
        )
    raise FileNotFoundError(f"No train.json, test.json, or val.json found in {root}")


def load_splits(
    datasets: Iterable[str] | None = None,
    annotation_roots: Iterable[str | Path] | None = None,
    auto_split: bool = False,
    auto_split_test_size: float = 0.2,
    seed: int = 42,
    max_samples: int | None = None,
) -> SplitRows:
    train_rows: list[DinoSample] = []
    test_rows: list[DinoSample] = []

    for dataset_name, root in resolve_annotation_roots(datasets, annotation_roots):
        split = _load_single_root(dataset_name, root, auto_split, auto_split_test_size, seed)
        train_rows.extend(split.train)
        test_rows.extend(split.test)

    if max_samples is not None:
        rng = random.Random(seed)
        train_rows = list(train_rows)
        test_rows = list(test_rows)
        rng.shuffle(train_rows)
        rng.shuffle(test_rows)
        train_rows = train_rows[:max_samples]
        test_rows = test_rows[:max_samples]

    return SplitRows(train=train_rows, test=test_rows)


class DinoFrameDataset(Dataset):
    def __init__(self, samples: list[DinoSample], processor):
        self.samples = samples
        self.processor = processor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str, str, str]:
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        label = torch.tensor(sample.label, dtype=torch.long)
        return pixel_values, label, sample.dataset, sample.video_id, sample.frame_id
