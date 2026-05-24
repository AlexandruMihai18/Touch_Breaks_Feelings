"""Annotate each touch sample with the centroid of its touch mask in normalized [0, 1] coords.

For every touch-positive sample, this script:
  1. Loads the touch mask from target_path / touch_mask_path.
  2. Finds all non-zero pixels (no blob filtering — matches DINO/Qwen training exactly).
  3. Computes the closest mask pixel to the mean of non-zero pixels.
  4. Normalizes by image dimensions → x_touch_centroid, y_touch_centroid ∈ [0, 1].

Non-touch samples and those with missing/empty masks get
x_touch_centroid=null, y_touch_centroid=null.

These coordinates match the exact GT point that DINO and Qwen models are trained to
predict, making them the correct reference for point-task RMSE evaluation.

Usage
-----
    python scripts/evaluation/touch_zones/annotate_touch_centroid.py \\
        data/greatest_hits/annotations/train.json \\
        data/epic_kitchen/annotations/train.json \\
        [--min-area 10] [--force] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _touch_mask_path(entry: dict) -> str | None:
    for key in ("touch_mask_path", "target_path"):
        val = entry.get(key)
        if val:
            return val
    return None


def _centroid_norm(mask_path: str, min_area: int) -> tuple[float, float] | None:
    """Return (x_norm, y_norm) of the mask pixel closest to the centroid, or None."""
    p = Path(mask_path)
    if not p.exists():
        return None
    mask = np.array(Image.open(p).convert("L"))
    ys, xs = np.nonzero(mask > 0)
    if len(xs) < min_area:
        return None
    h, w = mask.shape
    mean_x = xs.mean()
    mean_y = ys.mean()
    distances = (xs - mean_x) ** 2 + (ys - mean_y) ** 2
    idx = int(np.argmin(distances))
    return float(xs[idx]) / max(w - 1, 1), float(ys[idx]) / max(h - 1, 1)


def annotate_file(
    annotation_path: Path,
    min_area: int,
    force: bool,
    dry_run: bool,
) -> dict:
    with open(annotation_path) as f:
        annotations = json.load(f)

    counts = {"updated": 0, "skipped": 0, "already_done": 0}

    for entry in annotations:
        if not force and "x_touch_centroid" in entry:
            counts["already_done"] += 1
            continue

        if entry.get("type") != "touch":
            entry["x_touch_centroid"] = None
            entry["y_touch_centroid"] = None
            counts["skipped"] += 1
            continue

        mask_path = _touch_mask_path(entry)
        if not mask_path:
            entry["x_touch_centroid"] = None
            entry["y_touch_centroid"] = None
            counts["skipped"] += 1
            continue

        result = _centroid_norm(mask_path, min_area)
        if result is None:
            entry["x_touch_centroid"] = None
            entry["y_touch_centroid"] = None
            counts["skipped"] += 1
            continue

        entry["x_touch_centroid"] = result[0]
        entry["y_touch_centroid"] = result[1]
        counts["updated"] += 1

    if not dry_run:
        with open(annotation_path, "w") as f:
            json.dump(annotations, f, indent=2)

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("annotations", nargs="+", type=Path,
                        help="Annotation JSON files to update")
    parser.add_argument("--min-area", type=int, default=10, metavar="PX",
                        help="Minimum non-zero mask pixels to be considered valid (default: 10)")
    parser.add_argument("--force", action="store_true",
                        help="Re-compute even if x_touch_centroid is already set")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute but do not write changes to disk")
    args = parser.parse_args()

    for path in args.annotations:
        if not path.exists():
            print(f"[WARN] not found: {path}")
            continue
        counts = annotate_file(path, args.min_area, args.force, args.dry_run)
        tag = " (dry run)" if args.dry_run else ""
        print(
            f"{path}{tag}\n"
            f"  updated={counts['updated']}  skipped={counts['skipped']}  "
            f"already_done={counts['already_done']}"
        )


if __name__ == "__main__":
    main()
