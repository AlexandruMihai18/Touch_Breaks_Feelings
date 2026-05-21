#!/usr/bin/env python
"""Merge dataset.json files from all annotation folders into a single dataset.json."""

import json
import sys
from pathlib import Path


def merge_datasets(annotations_root: Path, output_path: Path):
    """Merge all dataset.json files from annotation subfolders."""
    datasets = []
    annotation_dirs = sorted(annotations_root.glob("*/"))

    if not annotation_dirs:
        print(f"No annotation folders found in {annotations_root}")
        return

    for ann_dir in annotation_dirs:
        dataset_file = ann_dir / "dataset.json"
        if dataset_file.exists():
            with open(dataset_file) as f:
                samples = json.load(f)
            datasets.extend(samples)
            print(f"✓ Loaded {len(samples)} samples from {ann_dir.name}")
        else:
            print(f"⚠ No dataset.json in {ann_dir.name}")

    with open(output_path, "w") as f:
        json.dump(datasets, f, indent=2)

    print(f"\n✓ Merged {len(datasets)} total samples into {output_path}")


if __name__ == "__main__":
    annotations_root = (
        Path(__file__).parents[2] / "data" / "manual_annotations" / "annotations"
    )
    output_path = (
        Path(__file__).parents[2]
        / "data"
        / "manual_annotations"
        / "annotations"
        / "merged_dataset.json"
    )

    if len(sys.argv) > 1:
        output_path = Path(sys.argv[1])

    merge_datasets(annotations_root, output_path)
