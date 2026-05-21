"""
Generate temporal context frames around annotated touch events.

Reads *_times.txt label files and split lists directly — no annotation JSONs
required — so this step can run right after frame extraction (step 2).

For every touch frame (material != None) in each video, samples 4 frames
before and 4 frames after the anchor, each 0.5 s apart:

    offsets: -4·Δ, -3·Δ, -2·Δ, -Δ, +Δ, +2·Δ, +3·Δ, +4·Δ  where Δ = --step-seconds (default 0.5 s)

Context frames are written to data/greatest_hits/frames/{video_id}/ alongside
the existing annotated frames.  Entries that land outside the video range or
coincide with an anchor frame are silently dropped.

Output (one file per split):
    data/greatest_hits/annotations/train_context_frames.json
    data/greatest_hits/annotations/val_context_frames.json

Each JSON entry:
    image_path       — absolute path to the extracted JPEG
    type             — "no-touch"
    video_id         — folder name (e.g. "2015-02-16-16-49-06_denoised")
    anchor_frame_idx — frame index of the related touch event
    offset_steps     — signed step number relative to anchor (-4…-1, 1…4)
    offset_s         — temporal offset in seconds (-2.0…-0.5, 0.5…2.0)

Usage
-----
    python scripts/greatest_hits/2_1_generate_context_frames.py

    python scripts/greatest_hits/2_1_generate_context_frames.py \\
        --data-dir   data/greatest_hits/vis-data-256/vis-data-256 \\
        --frames-dir data/greatest_hits/frames \\
        --output-dir data/greatest_hits/annotations \\
        --step-seconds 0.25 \\
        --overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parents[2]))
from inference_script.shared import GH_ANNO_DIR, GH_FRAMES_ROOT

_GH_RAW_DIR = GH_FRAMES_ROOT.parent / "vis-data-256" / "vis-data-256"

_STEPS = [-4, -3, -2, -1, 1, 2, 3, 4]
_DEFAULT_STEP_S = 0.5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract temporal context frames around touch events.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir",    type=Path, default=_GH_RAW_DIR,
                   help="Directory containing *_denoised.mp4 and *_times.txt files.")
    p.add_argument("--frames-dir",  type=Path, default=GH_FRAMES_ROOT,
                   help="Root directory for extracted frames.")
    p.add_argument("--output-dir",  type=Path, default=GH_ANNO_DIR,
                   help="Directory to write *_context_frames.json files.")
    p.add_argument("--train-split", type=Path, default=_GH_RAW_DIR / "train.txt",
                   help="Split txt file: train video IDs (one bare timestamp per line).")
    p.add_argument("--test-split",  type=Path, default=_GH_RAW_DIR / "test.txt",
                   help="Split txt file: test/val video IDs.")
    p.add_argument("--step-seconds", type=float, default=_DEFAULT_STEP_S,
                   help="Temporal gap between context steps in seconds.")
    p.add_argument("--image-quality", type=int, default=95,
                   help="JPEG quality for saved context frames.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-extract frames that already exist on disk.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Raw data helpers
# ---------------------------------------------------------------------------

def _load_split_ids(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    return {
        line.strip() + "_denoised"
        for line in path.read_text().splitlines()
        if line.strip()
    }


def _none_or_str(val: str) -> str | None:
    return None if val == "None" else val


def _touch_anchors_from_times(mp4: Path, times_path: Path) -> tuple[float, int, list[int]]:
    """Return (fps, total_frames, sorted touch frame indices) for one video."""
    cap = cv2.VideoCapture(str(mp4))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    seen: set[int] = set()
    anchors: list[int] = []
    for line in times_path.read_text().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        material = _none_or_str(parts[1]) if len(parts) > 1 else None
        if material is None:
            continue
        idx = min(round(float(parts[0]) * fps), max(total - 1, 0))
        if idx not in seen:
            seen.add(idx)
            anchors.append(idx)

    return fps, total, sorted(anchors)


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------

def _extract_frame(mp4: Path, frame_idx: int, out_path: Path, quality: int) -> bool:
    """Seek to frame_idx and write a JPEG. Returns True on success."""
    cap = cv2.VideoCapture(str(mp4))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return False
    cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return True


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def generate_context_entries(
    mp4: Path,
    video_id: str,
    fps: float,
    total: int,
    anchor_idxs: list[int],
    frames_dir: Path,
    step_seconds: float,
    image_quality: int,
    overwrite: bool,
) -> list[dict]:
    """Compute and extract context frames for one video. Returns annotation entries."""
    step_frames = round(step_seconds * fps)
    anchor_set = set(anchor_idxs)
    out_dir = frames_dir / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Map each unique context frame index to all (anchor, step) pairs referencing it.
    to_extract: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for anchor_idx in anchor_idxs:
        for step in _STEPS:
            ctx_idx = anchor_idx + step * step_frames
            if ctx_idx < 0 or ctx_idx >= total:
                continue
            if ctx_idx in anchor_set:
                continue
            to_extract[ctx_idx].append((anchor_idx, step))

    entries: list[dict] = []
    extracted = skipped = failed = 0

    for ctx_idx, references in sorted(to_extract.items()):
        out_path = out_dir / f"frame_{ctx_idx:06d}.jpg"
        if not out_path.exists() or overwrite:
            if not _extract_frame(mp4, ctx_idx, out_path, image_quality):
                failed += 1
                continue
            extracted += 1
        else:
            skipped += 1

        for anchor_idx, step in references:
            entries.append({
                "image_path":       str(out_path.resolve()),
                "type":             "no-touch",
                "video_id":         video_id,
                "anchor_frame_idx": anchor_idx,
                "offset_steps":     step,
                "offset_s":         round(step * step_seconds, 1),
            })

    print(f"  {video_id}: {len(anchor_idxs)} anchors → "
          f"{extracted} extracted, {skipped} skipped, {failed} failed")
    return entries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  Greatest Hits context frame generation")
    print(f"  data dir    : {args.data_dir}")
    print(f"  frames dir  : {args.frames_dir}")
    print(f"  output dir  : {args.output_dir}")
    print(f"  step size   : {args.step_seconds} s  (= ~{round(args.step_seconds * 30)} frames @ 30 fps)")
    print(f"  steps       : {_STEPS}")
    print(f"  overwrite   : {args.overwrite}")
    print("=" * 60)

    train_ids = _load_split_ids(args.train_split)
    test_ids  = _load_split_ids(args.test_split)
    if not train_ids and not test_ids:
        print("  Warning: no split files found — all videos go to train.")

    split_entries: dict[str, list[dict]] = {"train": [], "val": []}
    missing = 0

    for mp4 in sorted(args.data_dir.glob("*_denoised.mp4")):
        video_id = mp4.stem
        prefix = video_id.removesuffix("_denoised")
        times_path = args.data_dir / f"{prefix}_times.txt"
        if not times_path.exists():
            continue

        fps, total, anchor_idxs = _touch_anchors_from_times(mp4, times_path)
        if not anchor_idxs:
            continue

        entries = generate_context_entries(
            mp4=mp4,
            video_id=video_id,
            fps=fps,
            total=total,
            anchor_idxs=anchor_idxs,
            frames_dir=args.frames_dir,
            step_seconds=args.step_seconds,
            image_quality=args.image_quality,
            overwrite=args.overwrite,
        )

        if video_id in train_ids:
            split_entries["train"].extend(entries)
        elif video_id in test_ids:
            split_entries["val"].extend(entries)
        else:
            split_entries["train"].extend(entries)

        missing += sum(
            1 for e in entries
            if not Path(e["image_path"]).exists()
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, entries in split_entries.items():
        out_path = args.output_dir / f"{split}_context_frames.json"
        out_path.write_text(json.dumps(entries, indent=2))
        print(f"\n  {split}_context_frames.json → {len(entries):,} entries")

    print("\nDone.")


if __name__ == "__main__":
    main()
