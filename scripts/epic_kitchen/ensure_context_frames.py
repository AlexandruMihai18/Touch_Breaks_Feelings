"""
Verify that every ±4 context frame for EPIC Kitchen VAL touch anchors exists
on disk, and download any that are missing from the VISOR frame ZIPs.

After ensuring all frames are present, regenerates val_context_frames.json
(equivalent to re-running generate_context_frames.py --overwrite).

Frame ZIPs are fetched via HTTP range requests from:
    {FRAME_BASE}/val/{participant}/{video_id}.zip

Usage
-----
    python scripts/epic_kitchen/ensure_context_frames.py

    python scripts/epic_kitchen/ensure_context_frames.py \\
        --annotations-dir /scratch-shared/$USER/epic_kitchen/annotations \\
        --frames-dir      /scratch-shared/$USER/epic_kitchen/frames \\
        --workers         4
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
from scripts.epic_kitchen.download_epic_kitchen.constants import FRAME_BASE
from scripts.epic_kitchen.download_epic_kitchen.failure_log import FailureLog
from scripts.epic_kitchen.download_epic_kitchen.phase3_frames import (
    _extract_from_zip,
    _jpeg_ok,
)

_EK_FPS = 50
_STEPS = [-4, -3, -2, -1, 1, 2, 3, 4]
_DEFAULT_STEP_S = 0.5
_SPLIT = "val"

_FRAME_NUM_RE = re.compile(r"_frame_(\d+)\.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ensure EPIC Kitchen VAL context frames exist; download missing ones.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--annotations-dir", type=Path, default=EK_ANNO_DIR,
                   help="Directory containing val.json.")
    p.add_argument("--frames-dir", type=Path, default=EK_FRAMES_ROOT,
                   help="Root directory with per-video frame subfolders.")
    p.add_argument("--step-seconds", type=float, default=_DEFAULT_STEP_S,
                   help="Temporal gap between context steps in seconds.")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel threads for ZIP extraction.")
    p.add_argument("--failure-log", type=Path, default=None,
                   help="Path to failure log JSON (default: annotations-dir/ctx_failures.json).")
    return p.parse_args()


def _frame_idx_from_path(image_path: str) -> int | None:
    m = _FRAME_NUM_RE.search(Path(image_path).name)
    return int(m.group(1)) if m else None


def _frame_name(video_id: str, frame_idx: int) -> str:
    return f"{video_id}_frame_{frame_idx:010d}.jpg"


def _participant(video_id: str) -> str:
    return video_id.split("_")[0]


def _compute_needed(
    entries: list[dict],
    step_frames: int,
) -> dict[str, set[int]]:
    """Return {video_id: {context_frame_idx, ...}} for all VAL touch anchors."""
    touch_anchors: dict[str, set[int]] = defaultdict(set)
    for e in entries:
        if e.get("type") == "touch":
            idx = _frame_idx_from_path(e["image_path"])
            if idx is not None:
                touch_anchors[e["video_id"]].add(idx)

    needed: dict[str, set[int]] = {}
    for video_id, anchor_set in touch_anchors.items():
        ctx_idxs: set[int] = set()
        for anchor_idx in anchor_set:
            for step in _STEPS:
                ctx_idx = anchor_idx + step * step_frames
                if ctx_idx < 0 or ctx_idx in anchor_set:
                    continue
                ctx_idxs.add(ctx_idx)
        needed[video_id] = ctx_idxs

    return needed


def _build_context_entries(
    entries: list[dict],
    frames_dir: Path,
    step_frames: int,
    step_seconds: float,
) -> list[dict]:
    """Build val_context_frames.json entries from touch anchors."""
    touch_anchors: dict[str, set[int]] = defaultdict(set)
    for e in entries:
        if e.get("type") == "touch":
            idx = _frame_idx_from_path(e["image_path"])
            if idx is not None:
                touch_anchors[e["video_id"]].add(idx)

    context_entries: list[dict] = []
    for video_id, anchor_set in sorted(touch_anchors.items()):
        to_index: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for anchor_idx in sorted(anchor_set):
            for step in _STEPS:
                ctx_idx = anchor_idx + step * step_frames
                if ctx_idx < 0 or ctx_idx in anchor_set:
                    continue
                to_index[ctx_idx].append((anchor_idx, step))

        for ctx_idx, references in sorted(to_index.items()):
            frame_path = frames_dir / video_id / _frame_name(video_id, ctx_idx)
            if not _jpeg_ok(frame_path):
                continue
            for anchor_idx, step in references:
                context_entries.append({
                    "image_path":       str(frame_path.resolve()),
                    "type":             "no-touch",
                    "video_id":         video_id,
                    "anchor_frame_idx": anchor_idx,
                    "offset_steps":     step,
                    "offset_s":         round(step * step_seconds, 1),
                })

    return context_entries


def main() -> None:
    args = parse_args()
    flog_path = args.failure_log or args.annotations_dir / "ctx_failures.json"
    flog = FailureLog(flog_path)

    val_json = args.annotations_dir / "val.json"
    if not val_json.exists():
        print(f"ERROR: {val_json} not found.")
        sys.exit(1)

    step_frames = round(args.step_seconds * _EK_FPS)

    print("=" * 60)
    print("  EPIC Kitchen — ensure VAL context frames")
    print(f"  annotations : {val_json}")
    print(f"  frames dir  : {args.frames_dir}")
    print(f"  step size   : {args.step_seconds} s  (= {step_frames} frames @ {_EK_FPS} fps)")
    print(f"  steps       : {_STEPS}")
    print(f"  failure log : {flog_path}")
    print("=" * 60)

    entries: list[dict] = json.loads(val_json.read_text())
    needed = _compute_needed(entries, step_frames)

    total_needed = sum(len(v) for v in needed.values())
    print(f"\nContext frames needed: {total_needed} across {len(needed)} videos")

    # ── Check which are missing ──────────────────────────────────────────────
    missing_by_video: dict[str, set[str]] = {}
    total_ok = total_missing = 0

    for video_id, ctx_idxs in sorted(needed.items()):
        vid_dir = args.frames_dir / video_id
        missing: set[str] = set()
        for idx in ctx_idxs:
            name = _frame_name(video_id, idx)
            if not _jpeg_ok(vid_dir / name):
                missing.add(name)
        if missing:
            missing_by_video[video_id] = missing
            total_missing += len(missing)
        total_ok += len(ctx_idxs) - len(missing)

    print(f"  Already on disk : {total_ok}")
    print(f"  Missing         : {total_missing}")

    if not missing_by_video:
        print("\nAll context frames present.")
    else:
        print(f"\nDownloading {total_missing} missing frame(s) from VISOR ZIPs …\n")
        dl_ok = dl_fail = 0
        for video_id, missing_names in sorted(missing_by_video.items()):
            pid = _participant(video_id)
            zip_url = f"{FRAME_BASE}/{_SPLIT}/{pid}/{video_id}.zip"
            dest_dir = args.frames_dir / video_id
            print(f"  {video_id}: {len(missing_names)} missing → {zip_url}")
            ok, fail = _extract_from_zip(zip_url, missing_names, dest_dir, video_id, flog)
            dl_ok += ok
            dl_fail += fail

        print(f"\n  Downloaded: {dl_ok}  Failed: {dl_fail}")
        if dl_fail:
            print(f"  ⚠ {dl_fail} frame(s) could not be retrieved — see {flog_path}")
        flog.flush()

    # ── Regenerate val_context_frames.json ───────────────────────────────────
    print("\nRegenerating val_context_frames.json …")
    context_entries = _build_context_entries(entries, args.frames_dir, step_frames, args.step_seconds)
    out_path = args.annotations_dir / "val_context_frames.json"
    out_path.write_text(json.dumps(context_entries, indent=2))
    print(f"  val_context_frames.json → {len(context_entries):,} entries")

    print("\nDone.")


if __name__ == "__main__":
    main()
