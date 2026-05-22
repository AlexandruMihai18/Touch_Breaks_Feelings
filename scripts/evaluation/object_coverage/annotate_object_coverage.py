"""Annotate every sample with the fraction of the image covered by its object mask.

    object_coverage = nonzero pixels in object mask / total pixels

Works for both touch and no-touch entries across any dataset — all annotation
entries with an `object_mask_path` field are processed.  Entries without a
mask path or with a missing file get `object_coverage=null`.

Usage
-----
    python scripts/evaluation/object_coverage/annotate_object_coverage.py \\
        data/greatest_hits/annotations/train.json \\
        data/epic_kitchen/annotations/train.json \\
        [--force] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _compute_coverage(mask_path: str) -> float | None:
    p = Path(mask_path)
    if not p.exists():
        return None
    arr = np.array(Image.open(p).convert("L"))
    return float(np.count_nonzero(arr)) / arr.size


def annotate_file(path: Path, force: bool, dry_run: bool) -> dict:
    with open(path) as f:
        annotations = json.load(f)

    counts = {"updated": 0, "skipped": 0, "already_done": 0}

    for entry in annotations:
        if not force and "object_coverage" in entry:
            counts["already_done"] += 1
            continue

        mask_path = entry.get("object_mask_path")
        if not mask_path:
            entry["object_coverage"] = None
            counts["skipped"] += 1
            continue

        cov = _compute_coverage(mask_path)
        entry["object_coverage"] = cov
        if cov is not None:
            counts["updated"] += 1
        else:
            counts["skipped"] += 1

    if not dry_run:
        with open(path, "w") as f:
            json.dump(annotations, f, indent=2)

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("annotations", nargs="+", type=Path)
    parser.add_argument("--force",   action="store_true", help="Recompute even if object_coverage already set")
    parser.add_argument("--dry-run", action="store_true", help="Compute but do not write changes")
    args = parser.parse_args()

    for path in args.annotations:
        if not path.exists():
            print(f"[WARN] not found: {path}")
            continue
        counts = annotate_file(path, args.force, args.dry_run)
        tag = " (dry run)" if args.dry_run else ""
        print(
            f"{path}{tag}\n"
            f"  updated={counts['updated']}  skipped={counts['skipped']}  "
            f"already_done={counts['already_done']}"
        )


if __name__ == "__main__":
    main()
