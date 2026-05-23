#!/usr/bin/env python3
"""
Recompute Kubric touch masks from GT object masks and GT depth.

For each touch row, the original Kubric touch mask at target_path is preserved
as *_touch_gt.png and recorded in touch_gt_path. The active target_path is then
overwritten with a mask computed from object1_mask_path, object2_mask_path, and
depth_path using compute_touch_region_v2.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

_ANNO_DIR = _PROJECT_ROOT / "data" / "kubric_movi_a_256" / "annotations"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recompute Kubric touch masks from object masks and GT depth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--anno-dir",
        default=str(_ANNO_DIR),
        help="Directory containing split annotation JSON files.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        metavar="SPLIT",
        help="Annotation splits to process.",
    )
    parser.add_argument(
        "--dilation-radius",
        type=int,
        default=10,
        help="Contact zone dilation radius in pixels.",
    )
    parser.add_argument(
        "--abs-d-threshold",
        type=float,
        default=0.05,
        help="Max normalized local depth gap for contact pixels.",
    )
    parser.add_argument(
        "--local-radius",
        type=int,
        default=10,
        help="Local depth-estimation radius in pixels.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute target_path masks even if touch_gt_path already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report planned changes without writing masks or JSON.",
    )
    return parser.parse_args()


def _load_split(path: Path) -> list[dict] | None:
    if not path.exists():
        print(f"  Not found: {path} - skipping")
        return None
    return json.loads(path.read_text())


def _touch_gt_path(target_path: Path) -> Path:
    stem = target_path.stem
    if stem.endswith("_touch"):
        return target_path.with_name(
            stem.removesuffix("_touch") + "_touch_gt" + target_path.suffix
        )
    return target_path.with_name(stem + "_gt" + target_path.suffix)


def _load_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"))


def _load_depth(path: Path) -> np.ndarray:
    depth = np.array(Image.open(path))
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth.astype(np.float32)


def _is_touch_row(entry: dict) -> bool:
    return (
        entry.get("type") == "touch"
        or entry.get("sample_type") == "touch"
        or bool(entry.get("target_path"))
    )


def _missing_required(entry: dict) -> list[str]:
    required = ["target_path", "object1_mask_path", "object2_mask_path", "depth_path"]
    missing = []
    for key in required:
        value = entry.get(key)
        if not value:
            missing.append(key)
            continue
        if not Path(value).exists():
            missing.append(key)
    return missing


def _process_entry(entry: dict, args) -> tuple[str, int, int]:
    """Return (status, old_pixels, new_pixels)."""
    if not _is_touch_row(entry):
        return "non_touch", 0, 0

    missing = _missing_required(entry)
    if missing:
        return "missing", 0, 0

    target_path = Path(entry["target_path"])
    gt_path = Path(entry.get("touch_gt_path") or _touch_gt_path(target_path))
    entry["touch_gt_path"] = str(gt_path)

    should_recompute = args.overwrite or not gt_path.exists()

    if args.dry_run:
        return "would_write" if should_recompute else "skip", 0, 0

    gt_path.parent.mkdir(parents=True, exist_ok=True)
    if not gt_path.exists():
        shutil.copy2(target_path, gt_path)

    if not should_recompute:
        return "skip", 0, 0

    mask1 = _load_mask(Path(entry["object1_mask_path"]))
    mask2 = _load_mask(Path(entry["object2_mask_path"]))
    depth = _load_depth(Path(entry["depth_path"]))

    if mask1.shape != mask2.shape or mask1.shape != depth.shape[:2]:
        return "shape_error", 0, 0

    from touch_detection_alg.touch import compute_touch_region_v2

    old_pixels = int(np.count_nonzero(_load_mask(gt_path) > 0))
    new_mask = compute_touch_region_v2(
        mask1,
        mask2,
        depth,
        dilation_radius=args.dilation_radius,
        abs_d_threshold=args.abs_d_threshold,
        local_radius=args.local_radius,
    )
    new_pixels = int(np.count_nonzero(new_mask > 0))
    Image.fromarray(new_mask.astype(np.uint8)).save(target_path)
    return "written", old_pixels, new_pixels


def main():
    args = parse_args()
    anno_dir = Path(args.anno_dir)

    print("=" * 72)
    print("  Kubric GT-depth touch mask recompute")
    print(f"  anno dir      : {anno_dir}")
    print(f"  splits        : {args.splits}")
    print(f"  dilation      : {args.dilation_radius}px")
    print(f"  abs threshold : {args.abs_d_threshold}")
    print(f"  local radius  : {args.local_radius}px")
    print(f"  overwrite     : {args.overwrite}")
    print(f"  dry run       : {args.dry_run}")
    print("=" * 72)

    any_found = False
    for split in args.splits:
        anno_path = anno_dir / f"{split}.json"
        entries = _load_split(anno_path)
        if entries is None:
            continue
        any_found = True

        counts = {
            "written": 0,
            "would_write": 0,
            "skip": 0,
            "missing": 0,
            "shape_error": 0,
            "non_touch": 0,
        }
        old_pixel_total = 0
        new_pixel_total = 0
        empty_new_masks = 0

        print(f"\n  [{split}] {len(entries):,} rows")
        for i, entry in enumerate(entries, 1):
            status, old_pixels, new_pixels = _process_entry(entry, args)
            counts[status] = counts.get(status, 0) + 1
            old_pixel_total += old_pixels
            new_pixel_total += new_pixels
            if status == "written" and new_pixels == 0:
                empty_new_masks += 1

            if i % 100 == 0 or i == len(entries):
                print(
                    f"  {i}/{len(entries)}  written={counts['written']}  "
                    f"skip={counts['skip']}  missing={counts['missing']}  "
                    f"shape_err={counts['shape_error']}",
                    end="\r",
                    flush=True,
                )
        print()

        if not args.dry_run:
            anno_path.write_text(json.dumps(entries, indent=2))
            print(f"  Updated {anno_path.name} with touch_gt_path")

        print(
            "  "
            f"written={counts['written']}  "
            f"would_write={counts['would_write']}  "
            f"skip={counts['skip']}  "
            f"missing={counts['missing']}  "
            f"shape_error={counts['shape_error']}  "
            f"non_touch={counts['non_touch']}"
        )
        if counts["written"]:
            print(
                f"  old_touch_pixels={old_pixel_total:,}  "
                f"new_touch_pixels={new_pixel_total:,}  "
                f"empty_new_masks={empty_new_masks:,}"
            )

    if not any_found:
        print("No split JSON files found.")
        return

    if args.dry_run:
        print("\n[DRY RUN] No masks or JSON files were written.")


if __name__ == "__main__":
    main()
