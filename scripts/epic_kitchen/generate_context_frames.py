"""
Generate temporal context frames around annotated TOUCH events in the EPIC
Kitchen VAL set.

Reads val.json, finds every entry with type=="touch", and for each anchor
frame samples 4 frames before and 4 frames after (0.5 s apart at 50 fps):

    offsets: -4·Δ, -3·Δ, -2·Δ, -Δ, +Δ, +2·Δ, +3·Δ, +4·Δ  where Δ = --step-seconds (default 0.5 s)

Frames are already extracted on disk; this script only checks their existence
and builds the index — no video decoding required.

Context frames that do not exist on disk are silently skipped.

Output:
    {output-dir}/val_context_frames.json

Each JSON entry mirrors the Greatest Hits format:
    image_path       — absolute path to the JPEG
    type             — "no-touch"
    video_id         — e.g. "P02_03"
    anchor_frame_idx — frame index of the related touch event
    offset_steps     — signed step (-4…-1, 1…4)
    offset_s         — temporal offset in seconds (-2.0…-0.5, 0.5…2.0)

Usage
-----
    python scripts/epic_kitchen/generate_context_frames.py

    python scripts/epic_kitchen/generate_context_frames.py \\
        --annotations-dir data/epic_kitchen/annotations \\
        --frames-dir      data/epic_kitchen/frames \\
        --step-seconds    0.5 \\
        --overwrite
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from inference_script.shared import EK_ANNO_DIR, EK_FRAMES_ROOT

_EK_FPS = 50
_STEPS = [-4, -3, -2, -1, 1, 2, 3, 4]
_DEFAULT_STEP_S = 0.5

_FRAME_NUM_RE = re.compile(r"_frame_(\d+)\.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build val_context_frames.json for EPIC Kitchen.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--annotations-dir", type=Path, default=EK_ANNO_DIR,
                   help="Directory containing val.json.")
    p.add_argument("--frames-dir", type=Path, default=EK_FRAMES_ROOT,
                   help="Root directory with per-video frame subfolders.")
    p.add_argument("--step-seconds", type=float, default=_DEFAULT_STEP_S,
                   help="Temporal gap between context steps in seconds.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-write the output JSON even if it already exists.")
    return p.parse_args()


def _frame_idx_from_path(image_path: str) -> int | None:
    m = _FRAME_NUM_RE.search(Path(image_path).name)
    return int(m.group(1)) if m else None


def _frame_path(frames_dir: Path, video_id: str, frame_idx: int) -> Path:
    return frames_dir / video_id / f"{video_id}_frame_{frame_idx:010d}.jpg"


def main() -> None:
    args = parse_args()

    val_json = args.annotations_dir / "val.json"
    if not val_json.exists():
        print(f"ERROR: {val_json} not found.")
        sys.exit(1)

    out_path = args.annotations_dir / "val_context_frames.json"
    if out_path.exists() and not args.overwrite:
        print(f"  {out_path} already exists — use --overwrite to regenerate.")
        sys.exit(0)

    step_frames = round(args.step_seconds * _EK_FPS)

    print("=" * 60)
    print("  EPIC Kitchen context frame index")
    print(f"  annotations : {val_json}")
    print(f"  frames dir  : {args.frames_dir}")
    print(f"  step size   : {args.step_seconds} s  (= {step_frames} frames @ {_EK_FPS} fps)")
    print(f"  steps       : {_STEPS}")
    print("=" * 60)

    entries: list[dict] = json.loads(val_json.read_text())

    # Group touch frame indices by video.
    touch_anchors: dict[str, set[int]] = defaultdict(set)
    for e in entries:
        if e.get("type") != "touch":
            continue
        idx = _frame_idx_from_path(e["image_path"])
        if idx is not None:
            touch_anchors[e["video_id"]].add(idx)

    print(f"\nFound touch anchors in {len(touch_anchors)} videos.\n")

    context_entries: list[dict] = []

    for video_id, anchor_set in sorted(touch_anchors.items()):
        anchor_idxs = sorted(anchor_set)

        # Map each unique context frame index → all (anchor, step) pairs.
        to_index: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for anchor_idx in anchor_idxs:
            for step in _STEPS:
                ctx_idx = anchor_idx + step * step_frames
                if ctx_idx < 0:
                    continue
                if ctx_idx in anchor_set:
                    continue
                to_index[ctx_idx].append((anchor_idx, step))

        found = missing = 0
        for ctx_idx, references in sorted(to_index.items()):
            frame_path = _frame_path(args.frames_dir, video_id, ctx_idx)
            if not frame_path.exists():
                missing += 1
                continue
            found += 1
            for anchor_idx, step in references:
                context_entries.append({
                    "image_path":       str(frame_path.resolve()),
                    "type":             "no-touch",
                    "video_id":         video_id,
                    "anchor_frame_idx": anchor_idx,
                    "offset_steps":     step,
                    "offset_s":         round(step * args.step_seconds, 1),
                })

        print(f"  {video_id}: {len(anchor_idxs)} anchors → "
              f"{found} context frames found, {missing} missing")

    args.annotations_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(context_entries, indent=2))
    print(f"\n  val_context_frames.json → {len(context_entries):,} entries")
    print("\nDone.")


if __name__ == "__main__":
    main()
