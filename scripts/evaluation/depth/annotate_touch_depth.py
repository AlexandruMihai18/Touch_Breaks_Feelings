"""Annotate each touch sample with its median depth under the touch mask.

Looks for a depth file at `{image_stem}_depth.png` in the same directory as the
image.  Depth maps are colorized RGB PNGs produced by colorize_depth() using the
**inferno** colormap over a Depth-Anything-V2 disparity output:

    dark / purple (low grayscale) → far from camera
    bright / yellow  (high grayscale) → near / close to camera

They are converted to grayscale (L channel) to obtain a scalar depth proxy.
The median grayscale value under the touch-mask blob is stored as `depth_touch`
(float in [0, 255]).  No-touch samples and samples without a depth file get
`depth_touch=null`.

Usage
-----
    python scripts/annotate_touch_depth.py \\
        data/greatest_hits/annotations/train.json \\
        [--force] [--dry-run]
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


def median_depth_under_mask(depth_path: Path, mask_path: str) -> float | None:
    img = Image.open(depth_path)
    # Depth PNGs are saved as uint16 (values 0-65535). Pillow loads them in
    # mode "I" (32-bit int), and .convert("L") clamps to [0,255] instead of
    # rescaling — every value above ~1 becomes 255. Detect uint16 and rescale.
    if img.mode in ("I", "I;16", "I;16B"):
        depth = np.array(img, dtype=np.float32) * (255.0 / 65535.0)
    else:
        depth = np.array(img.convert("L"), dtype=np.float32)
    mask  = np.array(Image.open(mask_path).convert("L")) > 127
    if mask.sum() == 0:
        return None
    return float(np.median(depth[mask]))


def annotate_file(path: Path, force: bool, dry_run: bool) -> dict:
    with open(path) as f:
        annotations = json.load(f)

    counts = {"updated": 0, "skipped": 0, "no_depth": 0, "already_done": 0}

    for entry in annotations:
        if not force and "depth_touch" in entry:
            counts["already_done"] += 1
            continue

        if entry.get("type") != "touch":
            entry["depth_touch"] = None
            counts["skipped"] += 1
            continue

        dp = entry.get("depth_path", None)
        if dp is None:
            entry["depth_touch"] = None
            counts["no_depth"] += 1
            continue

        mask_path = _touch_mask_path(entry)
        if not mask_path:
            entry["depth_touch"] = None
            counts["skipped"] += 1
            continue

        val = median_depth_under_mask(dp, mask_path)
        entry["depth_touch"] = val
        if val is not None:
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
    parser.add_argument("--force",   action="store_true", help="Re-compute even if depth_touch already set")
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
            f"  updated={counts['updated']}  no_depth={counts['no_depth']}  "
            f"skipped={counts['skipped']}  already_done={counts['already_done']}"
        )


if __name__ == "__main__":
    main()
