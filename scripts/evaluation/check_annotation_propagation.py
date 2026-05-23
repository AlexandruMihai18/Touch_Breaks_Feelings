"""Check whether all annotation fields added by run_all_annotations.py are
present in every entry of every dataset split.

Fields checked (set by run_all_annotations.py):
  depth_touch      – float or null   (annotate_touch_depth)
  x_touch          – int or null     (annotate_touch_zones)
  y_touch          – int or null     (annotate_touch_zones)
  object_coverage  – float or null   (annotate_object_coverage)

Datasets checked (train + val):
  <data_root>/greatest_hits/annotations/
  <data_root>/epic_kitchen/annotations/

Usage
-----
    python scripts/evaluation/check_annotation_propagation.py --data-root /scratch-shared/$USER
    python scripts/evaluation/check_annotation_propagation.py --data-root /scratch-shared/$USER --verbose
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FIELDS = ["depth_touch", "x_touch", "y_touch", "object_coverage", "general_class", "gh_class"]

DATASET_SUBDIRS = {
    "gh": Path("greatest_hits/annotations"),
    "ek": Path("epic_kitchen/annotations"),
}

SPLITS = ["train", "val"]


def check_file(path: Path, verbose: bool) -> dict:
    entries = json.loads(path.read_text())
    n = len(entries)
    results: dict[str, int] = {}

    for field in FIELDS:
        missing = [i for i, e in enumerate(entries) if field not in e]
        results[field] = len(missing)
        if verbose and missing:
            print(f"      [{field}] missing in {len(missing)} entries "
                  f"(indices {missing[:5]}{'...' if len(missing) > 5 else ''})")

    return {"total": n, **results}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", type=Path, required=True, metavar="DIR",
                        help="Root directory containing dataset folders")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASET_SUBDIRS),
                        default=list(DATASET_SUBDIRS), metavar="DS",
                        help=f"Datasets to check: {list(DATASET_SUBDIRS)} (default: all)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show entry indices with missing fields")
    args = parser.parse_args()

    any_missing = False

    for ds_key in args.datasets:
        ds_dir = args.data_root / DATASET_SUBDIRS[ds_key]
        print(f"\n{'='*60}")
        print(f"  Dataset: {ds_key}  ({ds_dir})")
        print(f"{'='*60}")

        for split in SPLITS:
            path = ds_dir / f"{split}.json"
            if not path.exists():
                print(f"  [SKIP] not found: {path}")
                continue

            print(f"\n  {split}.json")
            result = check_file(path, args.verbose)
            n = result["total"]

            all_ok = True
            for field in FIELDS:
                missing = result[field]
                if missing == 0:
                    print(f"    {field:<20} ✓  all {n} entries")
                else:
                    pct = 100 * missing / n
                    print(f"    {field:<20} ✗  {missing}/{n} missing ({pct:.1f}%)")
                    all_ok = False
                    any_missing = True

            if all_ok:
                print(f"    → fully propagated")

    print(f"\n{'='*60}")
    if any_missing:
        print("  RESULT: some fields are missing — re-run run_all_annotations.py")
        print("    python scripts/evaluation/run_all_annotations.py --data-root <DIR>")
    else:
        print("  RESULT: all annotation fields present ✓")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
