"""
Build Greatest Hits ground-truth data in a single pass.

Reads *_times.txt labels and extracted frame paths, then writes:

  annotations/train.json       — training split
  annotations/val.json         — validation split (official test set)

Annotation entry fields:
    image_path   : absolute path to the JPEG frame
    audio_path   : absolute path to the denoised WAV for this video
    type         : "touch" | "no-touch"  (material is not None → touch)
    material     : material label or null
    action       : action label (e.g. "hit", "scratch") or null
    video_id     : folder name (e.g. "2015-03-28-19-34-13_denoised")
    frame_idx    : integer frame index
    timestamp_s  : event time in seconds within the video

Mask paths are absent at this stage — run generate_gh_mask_annotations.py
after 4_annotate_greatest_hits.py to add them.

Usage
-----
    python scripts/greatest_hits/generate_gh_gt_annotations.py

    python scripts/greatest_hits/generate_gh_gt_annotations.py \\
        --data-dir    data/greatest_hits/vis-data-256/vis-data-256 \\
        --frames-dir  data/greatest_hits/frames \\
        --train-split data/greatest_hits/vis-data-256/vis-data-256/train.txt \\
        --test-split  data/greatest_hits/vis-data-256/vis-data-256/test.txt \\
        --output-dir  data/greatest_hits/annotations
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parents[2]))
from inference_script.shared import GH_FRAMES_ROOT, GH_MASKS_ROOT

_GH_RAW_DIR      = GH_FRAMES_ROOT.parent / "vis-data-256" / "vis-data-256"
_ANNOTATIONS_DIR = GH_MASKS_ROOT.parent / "annotations"


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _none_or_str(val: str) -> str | None:
    return None if val == "None" else val


def _parse_times(times_path: Path) -> list[dict]:
    rows = []
    for line in times_path.read_text().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        rows.append({
            "timestamp_s": float(parts[0]),
            "material": _none_or_str(parts[1]) if len(parts) > 1 else None,
            "action":   _none_or_str(parts[2]) if len(parts) > 2 else None,
        })
    return rows


def _fps_and_total(video_path: Path) -> tuple[float, int]:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total


def _load_split_ids(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    return {
        line.strip() + "_denoised"
        for line in path.read_text().splitlines()
        if line.strip()
    }


# ---------------------------------------------------------------------------
# Core builder — single pass over the raw data
# ---------------------------------------------------------------------------

def build(data_dir: Path, frames_dir: Path) -> list[dict]:
    """Return annotation entries in a single pass.

    Each entry is ready for train/val JSON: image_path, type, material, video_id, frame_idx.
    """
    anno_entries: list[dict] = []
    missing = 0

    for mp4 in sorted(data_dir.glob("*_denoised.mp4")):
        prefix = mp4.stem[: -len("_denoised")]
        times_path = data_dir / f"{prefix}_times.txt"
        if not times_path.exists():
            continue

        video_id = mp4.stem
        frame_dir = frames_dir / video_id
        fps, total = _fps_and_total(mp4)
        annotations = _parse_times(times_path)
        audio_path = data_dir / f"{prefix}_denoised.wav"
        audio_str = str(audio_path.resolve()) if audio_path.exists() else None

        seen: set[int] = set()
        for ann in annotations:
            idx = min(round(ann["timestamp_s"] * fps), max(total - 1, 0))
            if idx in seen:
                continue
            seen.add(idx)

            frame_path = frame_dir / f"frame_{idx:06d}.jpg"
            if not frame_path.exists():
                missing += 1
                continue

            anno_entries.append({
                "image_path":  str(frame_path.resolve()),
                "audio_path":  audio_str,
                "type":        "touch" if ann["material"] is not None else "no-touch",
                "material":    ann["material"],
                "action":      ann["action"],
                "video_id":    video_id,
                "frame_idx":   idx,
                "timestamp_s": ann["timestamp_s"],
            })

    if missing:
        print(f"  Warning: {missing} annotated frame(s) not found on disk "
              f"(run 2_extract_greatest_hits_frames.py first).")

    return anno_entries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build GT annotation JSONs (train.json + val.json) in a single pass.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir",    type=Path, default=_GH_RAW_DIR,
                   help="Directory containing *_denoised.mp4 and *_times.txt files.")
    p.add_argument("--frames-dir",  type=Path, default=GH_FRAMES_ROOT,
                   help="Root directory written by 2_extract_greatest_hits_frames.py.")
    p.add_argument("--train-split", type=Path, default=_GH_RAW_DIR / "train.txt",
                   help="Split txt file: train video IDs (one bare timestamp per line).")
    p.add_argument("--test-split",  type=Path, default=_GH_RAW_DIR / "test.txt",
                   help="Split txt file: test/val video IDs.")
    p.add_argument("--output-dir",  type=Path, default=_ANNOTATIONS_DIR,
                   help="Directory to write train.json and val.json.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  Greatest Hits GT data generation")
    print(f"  data dir    : {args.data_dir}")
    print(f"  frames dir  : {args.frames_dir}")
    print(f"  output dir  : {args.output_dir}")
    print("=" * 60)

    train_ids = _load_split_ids(args.train_split)
    test_ids  = _load_split_ids(args.test_split)
    if not train_ids and not test_ids:
        print("  Warning: no split files found — all entries go to train.")
    else:
        print(f"  Splits: train={len(train_ids)} videos, test={len(test_ids)} videos")

    print("  Building from *_times.txt files...")
    anno_entries = build(args.data_dir, args.frames_dir)

    if not anno_entries:
        print("No entries found. Check --data-dir and --frames-dir.")
        sys.exit(1)

    n_touch    = sum(1 for e in anno_entries if e["type"] == "touch")
    n_no_touch = len(anno_entries) - n_touch
    print(f"  Total: {len(anno_entries):,}  ({n_touch:,} touch, {n_no_touch:,} no-touch)")

    # ── train / val splits ───────────────────────────────────────────────────
    train_entries: list[dict] = []
    val_entries:   list[dict] = []
    unassigned:    list[dict] = []

    for e in anno_entries:
        vid = e["video_id"]
        if vid in train_ids:
            train_entries.append(e)
        elif vid in test_ids:
            val_entries.append(e)
        else:
            unassigned.append(e)

    if unassigned:
        uniq = sorted({e["video_id"] for e in unassigned})
        print(f"  {len(unassigned)} entries from {len(uniq)} unassigned video(s) → placed in train")
        train_entries.extend(unassigned)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, entries in (("train", train_entries), ("val", val_entries)):
        out = args.output_dir / f"{split_name}.json"
        out.write_text(json.dumps(entries, indent=2))
        n_t  = sum(1 for e in entries if e["type"] == "touch")
        n_nt = len(entries) - n_t
        print(f"  {split_name}.json     → {out.name}  "
              f"({len(entries):,} entries, {n_t:,} touch, {n_nt:,} no-touch)")

    print("\nNext: run 4_annotate_greatest_hits.py, then 5_generate_gh_mask_annotations.py.")


if __name__ == "__main__":
    main()
