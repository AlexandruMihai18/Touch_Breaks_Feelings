"""Run all additional annotation steps for every dataset split.

Applies in order:
  1. annotate_touch_depth      → depth_touch
  2. annotate_touch_zones      → x_touch, y_touch
  3. annotate_object_coverage  → object_coverage
                                   (Kubric also gets object1_coverage,
                                    object2_coverage when masks are present)
  4. annotate_object_classes   → general_class, gh_class
     (only when --clusters-dir is provided or cluster JSONs are found under
      <data_root>/epic_kitchen/)

Datasets (train + val splits):
  - <data_root>/greatest_hits/annotations/
  - <data_root>/epic_kitchen/annotations/
  - <data_root>/kubric/annotations/

--data-root defaults to the repo's data/ directory; override to use scratch:
    --data-root /scratch-shared/$USER

Usage
-----
    python scripts/evaluation/run_all_annotations.py [--force] [--dry-run]
    python scripts/evaluation/run_all_annotations.py --datasets gh ek kubric [--force]
    python scripts/evaluation/run_all_annotations.py --data-root /scratch-shared/$USER
    python scripts/evaluation/run_all_annotations.py --data-root /scratch-shared/$USER \\
        --clusters-dir /scratch-shared/$USER/epic_kitchen
"""

from __future__ import annotations

import argparse
from pathlib import Path

from depth.annotate_touch_depth import annotate_file as annotate_depth
from touch_zones.annotate_touch_zones import annotate_file as annotate_zones
from object_coverage.annotate_object_coverage import annotate_file as annotate_coverage
from object_classes.annotate_object_classes import annotate_file as annotate_classes

DATASET_SUBDIRS = {
    "gh": Path("greatest_hits/annotations"),
    "ek": Path("epic_kitchen/annotations"),
    "kubric": Path("kubric/annotations"),
}

SPLITS = ["train.json", "val.json"]


def _header(label: str) -> None:
    print(f"\n{'='*60}\n  {label}\n{'='*60}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASET_SUBDIRS),
        default=list(DATASET_SUBDIRS),
        metavar="DS",
        help=f"Datasets to annotate: {list(DATASET_SUBDIRS)} (default: all)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        metavar="DIR",
        help="Root directory containing dataset folders (default: repo data/)",
    )
    parser.add_argument("--grid", type=int, default=8, metavar="N",
                        help="Grid dimension for touch zones (default: 8)")
    parser.add_argument("--min-blob-area", type=int, default=100, metavar="PX",
                        help="Minimum blob area for touch zones (default: 100)")
    parser.add_argument("--clusters-dir", type=Path, default=None, metavar="DIR",
                        help="Directory containing object_clusters_7.json and "
                             "object_clusters_gh.json (default: <data_root>/epic_kitchen/). "
                             "Object-class annotation is skipped if the files are not found.")
    parser.add_argument("--splits", nargs="+", default=["train.json", "val.json"],
                        metavar="SPLIT",
                        help="JSON files to process (default: train.json val.json)")
    parser.add_argument("--force", action="store_true",
                        help="Re-compute even if fields already set")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute but do not write changes")
    args = parser.parse_args()

    data_root = args.data_root if args.data_root else Path(__file__).resolve().parents[2] / "data"

    clusters_dir = args.clusters_dir if args.clusters_dir else data_root / "epic_kitchen"
    path_7  = clusters_dir / "object_clusters_7.json"
    path_gh = clusters_dir / "object_clusters_gh.json"
    if path_7.exists() and path_gh.exists():
        import json as _json
        _clusters_7  = _json.loads(path_7.read_text())
        _clusters_gh = _json.loads(path_gh.read_text())
        _classes_fn  = lambda p: annotate_classes(p, _clusters_7, _clusters_gh, args.force, args.dry_run)
        print(f"Object-class clusters loaded from {clusters_dir}")
    else:
        _classes_fn = None
        print(f"[SKIP] object-class annotation — cluster JSONs not found in {clusters_dir}")

    for ds_key in args.datasets:
        ds_dir = data_root / DATASET_SUBDIRS[ds_key]
        _header(f"Dataset: {ds_key}  ({ds_dir})")

        annotators = [
            ("depth_touch",      lambda p: annotate_depth(p, args.force, args.dry_run)),
            ("x/y_touch",        lambda p: annotate_zones(p, args.grid, args.min_blob_area, args.force, args.dry_run)),
            ("object_coverage",  lambda p: annotate_coverage(p, args.force, args.dry_run)),
        ]
        if ds_key != "kubric" and _classes_fn:
            annotators.append(("general/gh_class", _classes_fn))
        elif ds_key == "kubric":
            print("  [SKIP] object-class annotation — not applicable to Kubric")

        for split in args.splits:
            path = ds_dir / split
            if not path.exists():
                print(f"  [SKIP] not found: {path}")
                continue

            print(f"\n  Split: {split}")
            for label, fn in annotators:
                counts = fn(path)
                tag = " (dry run)" if args.dry_run else ""
                print(
                    f"    [{label}]{tag}  "
                    + "  ".join(f"{k}={v}" for k, v in counts.items())
                )


if __name__ == "__main__":
    main()
