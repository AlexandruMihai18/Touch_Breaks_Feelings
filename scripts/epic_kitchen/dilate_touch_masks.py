#!/usr/bin/env python3
"""
dilate_touch_masks.py — generate dilated touch-mask variants for EPIC Kitchen.

Dilates an existing touch mask by a given radius and clips the result to the
union of the hand and object masks.  This constrains expansion to pixels that
actually belong to either interacting object, making the label semantically
meaningful even at larger radii.

The output is written as  <stem>_dilated_r{radius}.png  beside the source file,
so nothing is ever overwritten.  Multiple radii can be processed in one run.

Source mask selection:
  By default the original  *_touch.png  is used.  Pass --input-suffix to dilate
  a different variant, e.g. --input-suffix touch_refined  to create
  *_touch_refined_dilated_r20.png from the depth-refined masks.

Usage:
    python scripts/epic_kitchen/dilate_touch_masks.py --radii 15 20 30
    python scripts/epic_kitchen/dilate_touch_masks.py --input-suffix touch_refined --radii 20
    python scripts/epic_kitchen/dilate_touch_masks.py --dry-run --radii 20
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

_ANNO_DIR = _PROJECT_ROOT / "data" / "epic_kitchen" / "annotations"


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate constrained-dilated touch-mask variants.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--anno-dir",
        default=str(_ANNO_DIR),
        help="Directory containing train.json / val.json annotation files.",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        metavar="SPLIT",
    )
    p.add_argument(
        "--radii",
        nargs="+",
        type=int,
        default=[20],
        metavar="R",
        help="One or more dilation radii (px) to generate.",
    )
    p.add_argument(
        "--input-suffix",
        default="touch",
        metavar="SUFFIX",
        help=(
            "Stem suffix of the source mask to dilate. "
            "'touch' = original *_touch.png; "
            "'touch_refined' = depth-refined masks."
        ),
    )
    p.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel I/O threads.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be written without writing anything.",
    )
    return p.parse_args()


def _load_entries(anno_dir: Path, splits: list[str]) -> list[dict]:
    entries = []
    for split in splits:
        path = anno_dir / f"{split}.json"
        if not path.exists():
            print(f"  ⚠ Not found: {path} — skipping")
            continue
        loaded = json.loads(path.read_text())
        for e in loaded:
            if "target_path" in e and "touch_mask_path" not in e:
                e["touch_mask_path"] = e["target_path"]
        print(f"  {split}: {len(loaded):,} entries")
        entries.extend(loaded)
    return entries


def _source_path(entry: dict, input_suffix: str) -> Path | None:
    """Resolve the source mask path for the given input_suffix."""
    base = Path(entry.get("touch_mask_path", ""))
    if not base.name:
        return None
    # target_path stem ends in '_touch'; replace that tail with the requested suffix.
    stem = base.stem  # e.g. P01_05_frame_..._p0_touch
    if input_suffix == "touch":
        return base  # original
    # For any other suffix: look for <dir>/<stem_without_touch>_<suffix>.png
    # The stem always ends in '_touch'; replace that part.
    if stem.endswith("_touch"):
        new_stem = stem[: -len("_touch")] + "_" + input_suffix
    else:
        new_stem = stem + "_" + input_suffix
    return base.parent / (new_stem + base.suffix)


def _output_path(source: Path, input_suffix: str, radius: int) -> Path:
    """Derive output path: append _dilated_r{radius} to the source stem."""
    return source.parent / (source.stem + f"_dilated_r{radius}" + source.suffix)


def _process_one(entry: dict, input_suffix: str, radius: int, overwrite: bool, dry_run: bool) -> str:
    src = _source_path(entry, input_suffix)
    if src is None or not src.exists():
        return "missing"

    out = _output_path(src, input_suffix, radius)
    if out.exists() and not overwrite:
        return "skip"
    if dry_run:
        return "ok"

    try:
        touch = np.array(Image.open(src).convert("L"))
        hand  = np.array(Image.open(entry["hand_mask_path"]).convert("L"))
        obj   = np.array(Image.open(entry["object_mask_path"]).convert("L"))
    except Exception as exc:
        print(f"\n  ⚠ Load error: {exc}")
        return "error"

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    dilated = cv2.dilate(touch, kernel)

    # Constrain to the union of hand and object pixels only.
    allowed = (hand > 0) | (obj > 0)
    result = np.where(allowed, dilated, 0).astype(np.uint8)

    Image.fromarray(result).save(out)
    return "ok"


def main():
    args = parse_args()
    anno_dir = Path(args.anno_dir)

    print("=" * 60)
    print("  Constrained touch mask dilation")
    print(f"  anno dir      : {anno_dir}")
    print(f"  splits        : {args.splits}")
    print(f"  input suffix  : {args.input_suffix}")
    print(f"  radii         : {args.radii} px")
    print(f"  workers       : {args.workers}")
    print(f"  overwrite     : {args.overwrite}")
    print(f"  dry run       : {args.dry_run}")
    print("=" * 60)

    entries = _load_entries(anno_dir, args.splits)
    if not entries:
        print("No entries found — nothing to do.")
        return

    for radius in args.radii:
        print(f"\n── radius = {radius} px ──────────────────────────────")
        ok = skip = missing = errors = 0
        total = len(entries)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {
                ex.submit(_process_one, e, args.input_suffix, radius, args.overwrite, args.dry_run): e
                for e in entries
            }
            for i, fut in enumerate(as_completed(futs), 1):
                result = fut.result()
                if result == "ok":       ok      += 1
                elif result == "skip":   skip    += 1
                elif result == "missing": missing += 1
                else:                    errors  += 1
                if i % 50 == 0 or i == total:
                    print(
                        f"  {i}/{total}  ✓{ok}  skip={skip}  miss={missing}  err={errors}",
                        end="\r", flush=True,
                    )

        print()
        if args.dry_run:
            print(f"  [DRY RUN] Would write : {ok}")
        else:
            print(f"  Written   : {ok}")
        print(f"  Skipped   : {skip}")
        if missing:
            print(f"  Missing   : {missing}  (source mask not found)")
        if errors:
            print(f"  Errors    : {errors}")

    print("\nDone.")


if __name__ == "__main__":
    main()
