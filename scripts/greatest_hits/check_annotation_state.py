#!/usr/bin/env python3
"""
check_annotation_state.py — assess the state of the Greatest Hits annotation job.

For every video folder under frames_dir, reports whether it is:
  complete  — masks/<video_id>/dataset.json exists (annotate_folder finished)
  partial   — masks/<video_id>/ exists but no dataset.json (failed mid-run)
  missing   — no masks/<video_id>/ directory at all

For complete folders it also checks frame coverage: how many frames in
frames/<video_id>/ have a matching *_touch.png in masks/<video_id>/.

Usage:
    python scripts/greatest_hits/check_annotation_state.py
    python scripts/greatest_hits/check_annotation_state.py --list partial
    python scripts/greatest_hits/check_annotation_state.py --list missing
"""

import argparse
import json
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

_GH_FRAMES_DEFAULT = _PROJECT_ROOT / "data" / "greatest_hits" / "frames"
_GH_MASKS_DEFAULT  = _PROJECT_ROOT / "data" / "greatest_hits" / "masks"


def parse_args():
    p = argparse.ArgumentParser(
        description="Assess Greatest Hits annotation job state.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--frames-dir", type=Path, default=_GH_FRAMES_DEFAULT)
    p.add_argument("--masks-dir",  type=Path, default=_GH_MASKS_DEFAULT)
    p.add_argument(
        "--list",
        choices=["complete", "partial", "missing", "all"],
        default=None,
        help="Print the folder names in that state (omit for summary only).",
    )
    return p.parse_args()


def _frame_count(folder: Path) -> int:
    return len(list(folder.glob("*.jpg"))) + len(list(folder.glob("*.png")))


def _touch_png_count(masks_folder: Path) -> int:
    return len(list(masks_folder.glob("*_touch.png")))


def _annotated_count(masks_folder: Path) -> int:
    ds = masks_folder / "dataset.json"
    if not ds.exists():
        return 0
    try:
        return len(json.loads(ds.read_text()))
    except Exception:
        return -1  # unreadable


def main():
    args = parse_args()

    if not args.frames_dir.exists():
        print(f"Frames directory not found: {args.frames_dir}")
        raise SystemExit(1)

    folders = sorted(d for d in args.frames_dir.iterdir() if d.is_dir())
    if not folders:
        print(f"No video folders found in {args.frames_dir}")
        raise SystemExit(0)

    complete = []
    partial  = []
    missing  = []

    total_frames     = 0
    total_annotated  = 0
    incomplete_touch = []  # complete folders where touch count < frame count

    for folder in folders:
        vid = folder.name
        masks_folder = args.masks_dir / vid
        n_frames = _frame_count(folder)
        total_frames += n_frames

        if (masks_folder / "dataset.json").exists():
            n_ann = _annotated_count(masks_folder)
            total_annotated += max(n_ann, 0)
            complete.append((vid, n_frames, n_ann))
            if n_ann >= 0 and n_ann < n_frames:
                incomplete_touch.append((vid, n_frames, n_ann))
        elif masks_folder.exists():
            n_pngs = _touch_png_count(masks_folder)
            partial.append((vid, n_frames, n_pngs))
        else:
            missing.append((vid, n_frames))

    print("=" * 60)
    print("  Greatest Hits annotation state")
    print(f"  frames dir : {args.frames_dir}")
    print(f"  masks dir  : {args.masks_dir}")
    print("=" * 60)
    print(f"\n  Total folders : {len(folders)}")
    print(f"  Complete      : {len(complete)}  ({len(complete)/len(folders)*100:.1f}%)")
    print(f"  Partial       : {len(partial)}   (masks dir exists, no dataset.json)")
    print(f"  Missing       : {len(missing)}   (not started)")
    print(f"\n  Total frames  : {total_frames:,}")
    print(f"  Annotated     : {total_annotated:,}  ({total_annotated/max(total_frames,1)*100:.1f}%)")

    if incomplete_touch:
        print(f"\n  ⚠  {len(incomplete_touch)} complete folders with fewer annotations than frames:")
        for vid, nf, na in incomplete_touch[:10]:
            print(f"     {vid}  {na}/{nf} frames annotated")
        if len(incomplete_touch) > 10:
            print(f"     … and {len(incomplete_touch) - 10} more")

    if args.list:
        print()
        if args.list in ("complete", "all"):
            print(f"  ── complete ({len(complete)}) ──")
            for vid, nf, na in complete:
                print(f"  {vid}  {na}/{nf}")
        if args.list in ("partial", "all"):
            print(f"\n  ── partial ({len(partial)}) ──")
            for vid, nf, np_ in partial:
                print(f"  {vid}  {np_} touch PNGs / {nf} frames")
        if args.list in ("missing", "all"):
            print(f"\n  ── missing ({len(missing)}) ──")
            for vid, nf in missing:
                print(f"  {vid}  ({nf} frames)")

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
