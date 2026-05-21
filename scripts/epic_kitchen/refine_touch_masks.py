#!/usr/bin/env python3
"""
refine_touch_masks.py — generate depth-refined touch masks for EPIC Kitchen.

For every annotated frame, runs:
  1. Depth-Anything-V2-Small to estimate monocular depth.
  2. Optional guided image filter to sharpen depth boundaries.
  3. compute_touch_region_v2 (pixel-level depth-gap filtering with
     connected-component recovery) to produce a refined touch mask.

The refined mask is written as  <stem>_touch_refined.png  beside the
original  <stem>_touch.png  — the original is never modified.

Usage:
    python scripts/epic_kitchen/refine_touch_masks.py
    python scripts/epic_kitchen/refine_touch_masks.py --splits train val --overwrite
    python scripts/epic_kitchen/refine_touch_masks.py --dry-run

Defaults match the tuned parameters from the depth-repair reviewer UI.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Allow imports from the project root regardless of CWD.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from touch_detection_alg.pipeline import annotate_touch

_ANNO_DIR = _PROJECT_ROOT / "data" / "epic_kitchen" / "annotations"


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate depth-refined touch masks for EPIC Kitchen annotations.",
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
        help="Which annotation splits to process.",
    )
    p.add_argument(
        "--dilation-radius",
        type=int,
        default=10,
        help="Contact zone dilation radius (px).",
    )
    p.add_argument(
        "--abs-d-threshold",
        type=float,
        default=0.05,
        help="Max absolute depth gap [0,1] to keep a contact pixel.",
    )
    p.add_argument(
        "--local-radius",
        type=int,
        default=10,
        help="Box-window half-width (px) for per-pixel local depth estimates.",
    )
    p.add_argument(
        "--guided-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sharpen depth with guided image filter before computing touch (default: on).",
    )
    p.add_argument(
        "--refine-radius",
        type=int,
        default=4,
        help="Guided-filter spatial radius (px).",
    )
    p.add_argument(
        "--refine-eps",
        type=float,
        default=0.1,
        help="Guided-filter regularisation ε.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing *_touch_refined.png files.",
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
        # Normalise field name: some files use target_path instead of touch_mask_path.
        for e in loaded:
            if "target_path" in e and "touch_mask_path" not in e:
                e["touch_mask_path"] = e["target_path"]
        print(f"  {split}: {len(loaded):,} entries")
        entries.extend(loaded)
    return entries


def _refined_path(touch_path: str) -> Path:
    p = Path(touch_path)
    return p.parent / (p.stem + "_refined" + p.suffix)


def _process_entry(entry: dict, args) -> str:
    """Run the refinement pipeline for one entry.  Returns 'ok', 'skip', or 'error'."""
    touch_src = entry.get("touch_mask_path", "")
    if not touch_src:
        return "error"

    out_path = _refined_path(touch_src)
    if out_path.exists() and not args.overwrite:
        return "skip"

    # Skip frames with no GT touch — this saves computation
    touch_arr = np.array(Image.open(touch_src).convert("L"))
    if not np.any(touch_arr > 0):
        if not args.dry_run:
            Image.fromarray(np.zeros_like(touch_arr)).save(out_path)
        return "ok"

    if args.dry_run:
        return "ok"

    try:
        img = np.array(Image.open(entry["image_path"]).convert("RGB"))
        hand = np.array(Image.open(entry["hand_mask_path"]).convert("L"))
        obj = np.array(Image.open(entry["object_mask_path"]).convert("L"))
    except Exception as exc:
        print(f"\n  ⚠ Load error {Path(entry['image_path']).name}: {exc}")
        return "error"

    try:
        refined = annotate_touch(
            img, hand, obj,
            dilation=args.dilation_radius,
            abs_d_threshold=args.abs_d_threshold,
            local_radius=args.local_radius,
            guided_filter=args.guided_filter,
            refine_radius=args.refine_radius,
            refine_eps=args.refine_eps,
        )
        Image.fromarray(refined).save(out_path)
    except Exception as exc:
        print(f"\n  ⚠ Refinement error {Path(entry['image_path']).name}: {exc}")
        return "error"

    return "ok"


def main():
    args = parse_args()
    anno_dir = Path(args.anno_dir)

    print("=" * 60)
    print("  Depth-refined touch mask generation")
    print(f"  anno dir       : {anno_dir}")
    print(f"  splits         : {args.splits}")
    print(f"  dilation       : {args.dilation_radius} px")
    print(f"  abs-d-thresh   : {args.abs_d_threshold}")
    print(f"  local radius   : {args.local_radius} px")
    print(f"  guided filter  : {args.guided_filter}"
          + (f"  (r={args.refine_radius}, ε={args.refine_eps})" if args.guided_filter else ""))
    print(f"  overwrite      : {args.overwrite}")
    print(f"  dry run        : {args.dry_run}")
    print("=" * 60)

    entries = _load_entries(anno_dir, args.splits)
    if not entries:
        print("No entries found — nothing to do.")
        return

    print(f"\n  Total entries  : {len(entries):,}")
    if not args.dry_run:
        print("  Loading depth model…")

    ok = skip = errors = 0
    total = len(entries)

    for i, entry in enumerate(entries, 1):
        result = _process_entry(entry, args)
        if result == "ok":
            ok += 1
        elif result == "skip":
            skip += 1
        else:
            errors += 1

        if i % 10 == 0 or i == total:
            print(
                f"  {i}/{total}  ✓{ok}  skip={skip}  err={errors}",
                end="\r",
                flush=True,
            )

    print()
    print("\n" + "=" * 60)
    if args.dry_run:
        print(f"  [DRY RUN] Would write : {ok}")
        print(f"  Already exist        : {skip}")
    else:
        print(f"  Written   : {ok}")
        print(f"  Skipped   : {skip}")
    if errors:
        print(f"  Errors    : {errors}")
    print("Done.")


if __name__ == "__main__":
    main()
