#!/usr/bin/env python3
"""
repair_depth_masks.py — recompute depth / refined-touch masks that are missing
or zero-byte (e.g. due to a prior disk-full failure).

For every annotation entry the script derives the expected output paths from
target_path (same convention as refine_touch_masks.py) and reprocesses only
entries whose depth PNG or refined-touch PNG is absent or empty.  Entries that
already have both valid files are silently skipped.

Usage:
    python scripts/epic_kitchen/repair_depth_masks.py
    python scripts/epic_kitchen/repair_depth_masks.py --splits train val --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from touch_detection_alg.pipeline import compute_depth
from touch_detection_alg.touch import compute_touch_region_v2

_ANNO_DIR = _PROJECT_ROOT / "data" / "epic_kitchen" / "annotations"


def parse_args():
    p = argparse.ArgumentParser(
        description="Repair missing / zero-byte depth and refined-touch masks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--anno-dir", default=str(_ANNO_DIR))
    p.add_argument("--splits", nargs="+", default=["train", "val"], metavar="SPLIT")
    p.add_argument("--dilation-radius", type=int, default=10)
    p.add_argument("--abs-d-threshold", type=float, default=0.05)
    p.add_argument("--local-radius", type=int, default=10)
    p.add_argument("--guided-filter", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--refine-radius", type=int, default=4)
    p.add_argument("--refine-eps", type=float, default=0.1)
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would be reprocessed without writing anything.")
    return p.parse_args()


def _touch_src(entry: dict) -> str:
    return entry.get("touch_mask_path") or entry.get("target_path", "")


def _depth_path(touch_path: str) -> Path:
    p = Path(touch_path)
    stem = p.stem.removesuffix("_touch")
    return p.parent / (stem + "_depth" + p.suffix)


def _refined_path(touch_path: str) -> Path:
    p = Path(touch_path)
    return p.parent / (p.stem + "_refined" + p.suffix)


def _valid(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _needs_repair(entry: dict) -> tuple[bool, bool]:
    """Return (need_depth, need_refined)."""
    src = _touch_src(entry)
    if not src:
        return False, False
    return not _valid(_depth_path(src)), not _valid(_refined_path(src))


def _repair_entry(entry: dict, args) -> str:
    src = _touch_src(entry)
    depth_out = _depth_path(src)
    refined_out = _refined_path(src)

    need_depth = not _valid(depth_out)
    need_refined = not _valid(refined_out)

    entry["depth_path"] = str(depth_out)

    if args.dry_run:
        return "ok"

    # ── Depth ──────────────────────────────────────────────────────────────────
    if need_depth:
        try:
            img = np.array(Image.open(entry["image_path"]).convert("RGB"))
            depth = compute_depth(
                img,
                guided_filter=args.guided_filter,
                refine_radius=args.refine_radius,
                refine_eps=args.refine_eps,
            )
            Image.fromarray((depth * 65535).astype(np.uint16)).save(depth_out)
        except Exception as exc:
            print(f"\n  ⚠ Depth error {Path(entry['image_path']).name}: {exc}")
            return "error"
    elif need_refined:
        # Depth file already valid — load it to avoid re-running the model.
        try:
            depth = np.array(Image.open(depth_out)).astype(np.float32) / 65535.0
        except Exception as exc:
            print(f"\n  ⚠ Load depth error {depth_out.name}: {exc}")
            return "error"

    # ── Refined touch mask ─────────────────────────────────────────────────────
    if need_refined:
        try:
            touch_arr = np.array(Image.open(src).convert("L"))
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
    print("  Depth mask repair")
    print(f"  anno dir  : {anno_dir}")
    print(f"  splits    : {args.splits}")
    print(f"  dry run   : {args.dry_run}")
    print("=" * 60)

    for split in args.splits:
        path = anno_dir / f"{split}.json"
        if not path.exists():
            print(f"\n  ⚠ Not found: {path} — skipping")
            continue

        entries = json.loads(path.read_text())
        for e in entries:
            if "target_path" in e and "touch_mask_path" not in e:
                e["touch_mask_path"] = e["target_path"]

        need_repair = [e for e in entries if any(_needs_repair(e))]
        print(f"\n  [{split}]  {len(entries):,} total  |  {len(need_repair):,} need repair")

        if not need_repair:
            print("  Nothing to do.")
            continue

        if args.dry_run:
            for e in need_repair[:5]:
                src = _touch_src(e)
                nd, nr = _needs_repair(e)
                flags = ("depth " if nd else "") + ("refined" if nr else "")
                print(f"    would repair [{flags.strip()}]: {Path(src).name}")
            if len(need_repair) > 5:
                print(f"    … and {len(need_repair) - 5} more")
            continue

        print("  Loading depth model…")
        ok = errors = 0
        total = len(need_repair)

        for i, entry in enumerate(need_repair, 1):
            result = _repair_entry(entry, args)
            if result == "ok":
                ok += 1
            else:
                errors += 1
            if i % 10 == 0 or i == total:
                print(f"  {i}/{total}  ✓{ok}  err={errors}", end="\r", flush=True)

        print()

        path.write_text(json.dumps(entries, indent=2))
        print(f"  ✎ Updated {path.name}")
        print(f"  Repaired={ok}  Errors={errors}")

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
