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

from touch_detection_alg.pipeline import compute_depth
from touch_detection_alg.touch import compute_touch_region_v2

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


def _load_split(anno_dir: Path, split: str) -> list[dict] | None:
    path = anno_dir / f"{split}.json"
    if not path.exists():
        print(f"  ⚠ Not found: {path} — skipping")
        return None
    entries = json.loads(path.read_text())
    for e in entries:
        if "target_path" in e and "touch_mask_path" not in e:
            e["touch_mask_path"] = e["target_path"]
    print(f"  {split}: {len(entries):,} entries")
    return entries


def _refined_path(touch_path: str) -> Path:
    p = Path(touch_path)
    return p.parent / (p.stem + "_refined" + p.suffix)


def _depth_path(touch_path: str) -> Path:
    p = Path(touch_path)
    stem = p.stem.removesuffix("_touch")
    return p.parent / (stem + "_depth" + p.suffix)


def _process_entry(entry: dict, args) -> str:
    """Run depth estimation + refinement for one entry.  Returns 'ok', 'skip', or 'error'.

    Depth is always computed (saved as uint16 PNG) so that downstream models
    can load it without re-running the depth network.  The refined touch mask
    reuses the same depth map rather than running the network twice.
    Side-effect: sets entry['depth_path'] before returning.
    """
    touch_src = entry.get("touch_mask_path", "")
    if not touch_src:
        return "error"

    refined_out = _refined_path(touch_src)
    depth_out = _depth_path(touch_src)

    entry["depth_path"] = str(depth_out)

    refined_done = refined_out.exists()
    depth_done = depth_out.exists()
    if refined_done and depth_done and not args.overwrite:
        return "skip"

    if args.dry_run:
        return "ok"

    try:
        img = np.array(Image.open(entry["image_path"]).convert("RGB"))
    except Exception as exc:
        print(f"\n  ⚠ Load error {Path(entry['image_path']).name}: {exc}")
        return "error"

    try:
        depth = compute_depth(
            img,
            guided_filter=args.guided_filter,
            refine_radius=args.refine_radius,
            refine_eps=args.refine_eps,
        )
        if not depth_done or args.overwrite:
            depth_uint16 = (depth * 65535).astype(np.uint16)
            Image.fromarray(depth_uint16).save(depth_out)
    except Exception as exc:
        print(f"\n  ⚠ Depth error {Path(entry['image_path']).name}: {exc}")
        return "error"

    if not refined_done or args.overwrite:
        try:
            touch_arr = np.array(Image.open(touch_src).convert("L"))
            if not np.any(touch_arr > 0):
                Image.fromarray(np.zeros_like(touch_arr)).save(refined_out)
            else:
                hand = np.array(Image.open(entry["hand_mask_path"]).convert("L"))
                obj = np.array(Image.open(entry["object_mask_path"]).convert("L"))
                refined = compute_touch_region_v2(
                    hand, obj, depth,
                    dilation_radius=args.dilation_radius,
                    abs_d_threshold=args.abs_d_threshold,
                    local_radius=args.local_radius,
                )
                Image.fromarray(refined).save(refined_out)
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

    any_found = False
    for split in args.splits:
        entries = _load_split(anno_dir, split)
        if entries is None:
            continue
        any_found = True

        print(f"\n  [{split}]  {len(entries):,} entries")
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

        if not args.dry_run:
            anno_path = anno_dir / f"{split}.json"
            anno_path.write_text(json.dumps(entries, indent=2))
            print(f"  ✎ Updated {anno_path.name} with depth_path field")

        print(f"  Written={ok}  Skipped={skip}  Errors={errors}")

    if not any_found:
        print("No entries found — nothing to do.")
        return

    print("\n" + "=" * 60)
    if args.dry_run:
        print("  [DRY RUN] No files written.")
    print("Done.")


if __name__ == "__main__":
    main()
