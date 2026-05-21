"""Assign each touch annotation to an N×N grid cell based on its largest touch blob.

For every touch-positive sample in one or more annotation JSON files, this script:
  1. Loads the touch mask.
  2. Finds the largest connected blob (to discard noise / false-positive fragments).
  3. Computes the blob centroid and maps it to a grid cell (x_touch, y_touch ∈ [0, N)).
  4. Writes x_touch and y_touch back into the annotation entry.

No-touch samples get x_touch=null, y_touch=null.

Usage
-----
    python scripts/annotate_touch_zones.py \\
        data/greatest_hits/annotations/train.json \\
        data/epic_kitchen/annotations/train.json \\
        [--grid 8] [--min-blob-area 100] [--force] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def _load_binary_mask(path: str) -> np.ndarray | None:
    p = Path(path)
    if not p.exists():
        return None
    arr = np.array(Image.open(p).convert("L"))
    return arr > 127


def _largest_blob_centroid(mask: np.ndarray, min_area: int) -> tuple[float, float] | None:
    """Return (row, col) centroid of the largest blob, or None if below min_area."""
    labeled, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    best_label = int(np.argmax(sizes)) + 1
    if sizes[best_label - 1] < min_area:
        return None
    return ndimage.center_of_mass(labeled == best_label)


def _centroid_to_cell(row: float, col: float, H: int, W: int, grid: int) -> tuple[int, int]:
    x = min(int(col / W * grid), grid - 1)
    y = min(int(row / H * grid), grid - 1)
    return x, y


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def _touch_mask_path(entry: dict) -> str | None:
    for key in ("touch_mask_path", "target_path"):
        val = entry.get(key)
        if val:
            return val
    return None


def annotate_file(
    annotation_path: Path,
    grid: int,
    min_blob_area: int,
    force: bool,
    dry_run: bool,
) -> dict:
    with open(annotation_path) as f:
        annotations = json.load(f)

    counts = {"updated": 0, "skipped": 0, "already_done": 0}

    for entry in annotations:
        if not force and "x_touch" in entry:
            counts["already_done"] += 1
            continue

        if entry.get("type") != "touch":
            entry["x_touch"] = None
            entry["y_touch"] = None
            counts["skipped"] += 1
            continue

        mask_path = _touch_mask_path(entry)
        if not mask_path:
            entry["x_touch"] = None
            entry["y_touch"] = None
            counts["skipped"] += 1
            continue

        mask = _load_binary_mask(mask_path)
        if mask is None:
            entry["x_touch"] = None
            entry["y_touch"] = None
            counts["skipped"] += 1
            continue

        H, W = mask.shape
        centroid = _largest_blob_centroid(mask, min_blob_area)
        if centroid is None:
            entry["x_touch"] = None
            entry["y_touch"] = None
            counts["skipped"] += 1
            continue

        x, y = _centroid_to_cell(centroid[0], centroid[1], H, W, grid)
        entry["x_touch"] = x
        entry["y_touch"] = y
        counts["updated"] += 1

    if not dry_run:
        with open(annotation_path, "w") as f:
            json.dump(annotations, f, indent=2)

    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("annotations", nargs="+", type=Path, help="Annotation JSON files to update")
    parser.add_argument("--grid", type=int, default=8, metavar="N",
                        help="Grid dimension — N×N cells (default: 8)")
    parser.add_argument("--min-blob-area", type=int, default=100, metavar="PX",
                        help="Minimum blob size in pixels to be considered a real touch (default: 100)")
    parser.add_argument("--force", action="store_true",
                        help="Re-compute even if x_touch/y_touch are already set")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute assignments but do not write changes to disk")
    args = parser.parse_args()

    for path in args.annotations:
        if not path.exists():
            print(f"[WARN] not found: {path}")
            continue
        counts = annotate_file(path, args.grid, args.min_blob_area, args.force, args.dry_run)
        tag = " (dry run)" if args.dry_run else ""
        print(
            f"{path}{tag}\n"
            f"  updated={counts['updated']}  skipped={counts['skipped']}  "
            f"already_done={counts['already_done']}"
        )


if __name__ == "__main__":
    main()
