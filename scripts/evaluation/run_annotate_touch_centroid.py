"""Populate x_touch_centroid / y_touch_centroid for all dataset annotation files.

Runs annotate_touch_centroid for all three datasets (greatest_hits, epic_kitchen,
kubric) across train and val splits.  The centroid is the mask pixel closest to the
mean of non-zero pixels, normalized to [0, 1] — matching the exact GT point that
DINO and Qwen models are trained to predict.

Usage
-----
    python scripts/evaluation/run_annotate_touch_centroid.py [--force] [--dry-run]
    python scripts/evaluation/run_annotate_touch_centroid.py --data-root /scratch-shared/$USER
    python scripts/evaluation/run_annotate_touch_centroid.py --datasets gh ek [--force]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from touch_zones.annotate_touch_centroid import annotate_file

DATASET_SUBDIRS = {
    "gh":     Path("greatest_hits/annotations"),
    "ek":     Path("epic_kitchen/annotations"),
    "kubric": Path("kubric/annotations"),
}

SPLITS = ["train.json", "val.json"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=list(DATASET_SUBDIRS),
        default=list(DATASET_SUBDIRS), metavar="DS",
        help="Datasets to annotate: gh ek kubric (default: all)",
    )
    parser.add_argument(
        "--data-root", type=Path, default=None, metavar="DIR",
        help="Root directory containing dataset folders (default: repo data/)",
    )
    parser.add_argument(
        "--splits", nargs="+", default=SPLITS, metavar="SPLIT",
        help="JSON splits to process (default: train.json val.json)",
    )
    parser.add_argument(
        "--min-area", type=int, default=10, metavar="PX",
        help="Minimum non-zero mask pixels to be considered valid (default: 10)",
    )
    parser.add_argument("--force", action="store_true",
                        help="Re-compute even if already annotated")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute but do not write changes to disk")
    args = parser.parse_args()

    data_root = args.data_root or Path(__file__).resolve().parents[2] / "data"

    for ds_key in args.datasets:
        ds_dir = data_root / DATASET_SUBDIRS[ds_key]
        print(f"\n{'=' * 60}\n  Dataset: {ds_key}  ({ds_dir})\n{'=' * 60}")
        for split in args.splits:
            path = ds_dir / split
            if not path.exists():
                print(f"  [SKIP] not found: {path}")
                continue
            counts = annotate_file(path, args.min_area, args.force, args.dry_run)
            tag = " (dry run)" if args.dry_run else ""
            print(
                f"  {split}{tag}  "
                + "  ".join(f"{k}={v}" for k, v in counts.items())
            )


if __name__ == "__main__":
    main()
