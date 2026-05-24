"""Annotate every sample with the fraction of the image covered by object masks.

    object_coverage = nonzero pixels in object mask / total pixels

Works for both touch and no-touch entries across any dataset — all annotation
entries with an `object_mask_path` field are processed.  Entries without a
mask path or with a missing file get `object_coverage=null`.

Kubric entries can use `object1_mask_path` and `object2_mask_path`; these get
`object1_coverage`, `object2_coverage`, and combined union `object_coverage`.

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


def _load_mask(mask_path: str) -> np.ndarray | None:
    p = Path(mask_path)
    if not p.exists():
        return None
    return np.array(Image.open(p).convert("L")) > 0


def _coverage_from_mask(mask: np.ndarray | None) -> float | None:
    if mask is None:
        return None
    return float(np.count_nonzero(mask)) / mask.size


def _compute_coverage(mask_path: str) -> float | None:
    arr = _load_mask(mask_path)
    return _coverage_from_mask(arr)


def _compute_kubric_coverages(entry: dict) -> tuple[float | None, float | None, float | None]:
    mask1_path = entry.get("object1_mask_path")
    mask2_path = entry.get("object2_mask_path")
    mask1 = _load_mask(mask1_path) if mask1_path else None
    mask2 = _load_mask(mask2_path) if mask2_path else None
    cov1 = _coverage_from_mask(mask1)
    cov2 = _coverage_from_mask(mask2)

    if mask1 is not None and mask2 is not None:
        if mask1.shape != mask2.shape:
            return cov1, cov2, None
        union = mask1 | mask2
    elif mask1 is not None:
        union = mask1
    elif mask2 is not None:
        union = mask2
    else:
        union = None
    return cov1, cov2, _coverage_from_mask(union)


def _kubric_done(entry: dict) -> bool:
    return all(k in entry for k in ("object1_coverage", "object2_coverage", "object_coverage"))


def _has_kubric_masks(entry: dict) -> bool:
    return bool(entry.get("object1_mask_path") or entry.get("object2_mask_path"))


def annotate_file(path: Path, force: bool, dry_run: bool) -> dict:
    with open(path) as f:
        annotations = json.load(f)

    counts = {"updated": 0, "skipped": 0, "already_done": 0}

    for entry in annotations:
        is_kubric = _has_kubric_masks(entry)
        if not force and ((is_kubric and _kubric_done(entry)) or (not is_kubric and "object_coverage" in entry)):
            counts["already_done"] += 1
            continue

        if is_kubric:
            cov1, cov2, cov = _compute_kubric_coverages(entry)
            entry["object1_coverage"] = cov1
            entry["object2_coverage"] = cov2
            entry["object_coverage"] = cov
            if cov1 is not None or cov2 is not None or cov is not None:
                counts["updated"] += 1
            else:
                counts["skipped"] += 1
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
