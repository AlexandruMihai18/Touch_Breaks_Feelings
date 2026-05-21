"""
Generate JSON annotation files for touch-region segmentation training.

Expected directory layout (produced by preprocess_frames.py + mask pipeline):

    frames_dir/
        video_001/
            frame_000042.jpg
            frame_000107.jpg
        video_002/
            ...
    masks_dir/
        video_001/
            frame_000042_hand.png     <- binary hand mask (0/255)
            frame_000042_object.png   <- binary object mask (0/255)
            frame_000042_touch.png    <- binary touch mask (0/255)
            frame_000107_hand.png
            ...
        video_002/
            ...

Each annotated triplet gets:
    {
        "image_path":        "<absolute path to frame>",
        "hand_mask_path":    "<absolute path to hand mask>",
        "object_mask_path":  "<absolute path to object mask>",
        "target_path":       "<absolute path to touch mask>",
        "type":              "touch" | "no-touch",
        "video_id":          "<folder name>"
    }

type is "touch" when the touch mask contains at least one non-zero pixel,
"no-touch" when it is entirely black.

Triplets are matched by: same video subfolder AND frame stem (e.g. "frame_000042"
maps to "frame_000042_hand.png", "frame_000042_object.png", "frame_000042_touch.png").
Videos with no mask subfolder are skipped with a warning.
Frames missing any of the three required masks are skipped.

Train/val split is stratified at the VIDEO × TYPE level: for each (video, type)
group, `val_frac` of its frames go to validation, so every video appears in both
splits and the touch/no-touch balance is preserved within each split.

Usage
-----
    python src/generate_annotations.py \\
        --frames_dir /data/frames \\
        --masks_dir  /data/masks  \\
        --output_dir /data/annotations \\
        --val_frac   0.15 \\
        --seed       42
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[3]))
from scripts.stratified_sampling import split_data

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MASK_EXTS = {".png", ".jpg", ".jpeg"}

# Matches: {frame_stem}_p{pair_idx}_{hand|object|touch}
_PAIR_MASK_RE = re.compile(r"^(.+)_p(\d+)_(hand|object|touch)$")

# EK100 frame names: {video_id}_frame_{NNNNNNNNNN}.jpg  (50 fps)
_FRAME_NUM_RE = re.compile(r"_frame_(\d+)\.")
_EPIC_FPS     = 50
_AUDIO_EXT    = "aac"


def _has_touch(touch_path: Path) -> bool:
    """Return True if the touch mask contains at least one non-zero pixel."""
    return Image.open(touch_path).convert("L").getbbox() is not None


def _mask_has_content(path: str) -> bool:
    try:
        return Image.open(path).convert("L").getbbox() is not None
    except Exception:
        return False


def _ctx_key(entry: dict) -> str:
    """Grouping key for context sampling — mirrors touch_dataset._context_key."""
    obj = entry.get("object_name", "").strip()
    return obj if obj else entry.get("video_id", "")


def _build_ctx_index(pairs: list[dict]) -> dict[str, list[int]]:
    """Build context lookup: key → [valid sample indices] (0-based into *pairs*).

    Only includes samples whose hand and object masks are non-empty, matching
    the validity filter applied at dataset load time.  Saving this alongside
    the annotation JSON lets TouchPairDataset skip the per-mask file scan on
    every run.
    """
    index: dict[str, list[int]] = {}
    for i, entry in enumerate(pairs):
        if not (_mask_has_content(entry.get("hand_mask_path", ""))
                and _mask_has_content(entry.get("object_mask_path", ""))):
            continue
        key = _ctx_key(entry)
        index.setdefault(key, []).append(i)
    return index


def _build_mask_triplets(
    mask_video_dir: Path,
) -> dict[tuple[str, int], dict[str, Path]]:
    """Return {(frame_stem, pair_idx): {hand: p, object: p, touch: p}} for all masks."""
    triplets: dict[tuple[str, int], dict[str, Path]] = {}
    for p in mask_video_dir.iterdir():
        if p.suffix.lower() not in MASK_EXTS:
            continue
        m = _PAIR_MASK_RE.match(p.stem)
        if m:
            base, pidx, mtype = m.group(1), int(m.group(2)), m.group(3)
            triplets.setdefault((base, pidx), {})[mtype] = p
    return triplets


def generate(
    frames_dir: str | Path,
    masks_dir: str | Path,
    output_dir: str | Path,
    val_frac: float = 0.15,
    seed: int = 42,
    audio_root: str | Path | None = None,
) -> None:
    frames_dir = Path(frames_dir).resolve()
    masks_dir = Path(masks_dir).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if audio_root is not None:
        audio_root = Path(audio_root)

    all_pairs: list[dict] = []
    skipped_no_mask_dir: list[str] = []
    skipped_incomplete: int = 0

    for video_dir in sorted(frames_dir.iterdir()):
        if not video_dir.is_dir():
            continue

        video_id = video_dir.name
        mask_video_dir = masks_dir / video_id

        if not mask_video_dir.exists():
            skipped_no_mask_dir.append(video_id)
            continue

        triplets = _build_mask_triplets(mask_video_dir)

        # Load optional per-pair metadata (object_name, label) written by phase4.
        meta_path = mask_video_dir / "pair_metadata.json"
        pair_meta: dict = json.loads(meta_path.read_text()) if meta_path.exists() else {}

        # Build a stem→path lookup once per video dir for O(1) access.
        frames_by_stem = {
            p.stem: p
            for p in video_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTS
        }

        for (base_stem, pair_idx), t in sorted(triplets.items()):
            frame_path = frames_by_stem.get(base_stem)
            if frame_path is None or not all(k in t for k in ("touch", "hand", "object")):
                skipped_incomplete += 1
                continue

            pair_key = f"{base_stem}_p{pair_idx}"
            object_name = (pair_meta.get(pair_key) or {}).get("object_name") or ""
            entry: dict = {
                "image_path": str(frame_path),
                "hand_mask_path": str(t["hand"]),
                "object_mask_path": str(t["object"]),
                "target_path": str(t["touch"]),
                "type": "touch" if _has_touch(t["touch"]) else "no-touch",
                "video_id": video_id,
                "object_name": object_name,
            }

            m_num = _FRAME_NUM_RE.search(frame_path.name)
            if m_num is not None:
                frame_idx = int(m_num.group(1))
                entry["frame_idx"] = frame_idx
                entry["audio_timestamp_sec"] = frame_idx / _EPIC_FPS
            if audio_root is not None:
                ap = audio_root / f"{video_id}.{_AUDIO_EXT}"
                entry["audio_path"] = str(ap) if ap.exists() else ""

            all_pairs.append(entry)

    n_videos = len({p["video_id"] for p in all_pairs})
    print(f"Found {len(all_pairs)} pairs across {n_videos} videos.")
    if skipped_no_mask_dir:
        print(
            f"  Skipped {len(skipped_no_mask_dir)} video(s) with no mask folder: "
            f"{skipped_no_mask_dir[:5]}{'...' if len(skipped_no_mask_dir) > 5 else ''}"
        )
    if skipped_incomplete:
        print(f"  Skipped {skipped_incomplete} frame(s) with incomplete mask triplet.")

    if not all_pairs:
        print("No pairs found — check your directory structure and extensions.")
        return

    # Stratified split by (video_id, type): preserves the touch/no-touch balance
    # within each video across both splits. Groups with only one sample are
    # dropped by split_data (can't be split).
    df = pd.DataFrame(all_pairs)
    train_df, val_df = split_data(
        df,
        train_size=1 - val_frac,
        strata=["video_id", "type"],
        random_state=seed,
    )
    train_pairs = train_df.to_dict(orient="records")
    val_pairs = val_df.to_dict(orient="records")

    for split, pairs in [("train", train_pairs), ("val", val_pairs)]:
        out = output_dir / f"{split}.json"
        out.write_text(json.dumps(pairs, indent=2))
        n_touch = sum(1 for p in pairs if p["type"] == "touch")
        n_no_touch = len(pairs) - n_touch
        print(
            f"  {split}: {len(pairs)} pairs "
            f"({n_touch} touch, {n_no_touch} no-touch, "
            f"{len({p['video_id'] for p in pairs})} videos) → {out}"
        )
        ctx_index = _build_ctx_index(pairs)
        ctx_out = output_dir / f"{split}_ctx_index.json"
        ctx_out.write_text(json.dumps(ctx_index))
        print(f"    ctx_index: {len(ctx_index)} groups → {ctx_out.name}")


def build_ctx_index_for_existing(annotations_dir: str | Path) -> None:
    """Write *_ctx_index.json for existing train.json / val.json in *annotations_dir*.

    Use this for datasets (e.g. manual_annotations) whose masks don't follow
    the paired-mask naming convention and therefore can't be re-generated via
    the full generate() pipeline.
    """
    annotations_dir = Path(annotations_dir)
    for split in ("train", "val"):
        json_path = annotations_dir / f"{split}.json"
        if not json_path.exists():
            continue
        pairs = json.loads(json_path.read_text())
        ctx_index = _build_ctx_index(pairs)
        out = annotations_dir / f"{split}_ctx_index.json"
        out.write_text(json.dumps(ctx_index))
        print(f"  {split}_ctx_index.json: {len(ctx_index)} groups → {out}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build train/val JSON annotation files for touch-region training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--frames_dir", default=None, help="Root dir with per-video frame subfolders")
    p.add_argument("--masks_dir",  default=None, help="Root dir with per-video mask subfolders")
    p.add_argument("--output_dir", default=None, help="Where to write train.json and val.json")
    p.add_argument("--audio_root", default=None,
                   help="Root dir with audio/{video_id}.aac files; adds audio_path + "
                        "audio_timestamp_sec fields to annotation entries")
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--ctx-only",
        metavar="ANNOTATIONS_DIR",
        default=None,
        help="Skip full generation; just write *_ctx_index.json for existing "
             "train.json / val.json in the given directory.",
    )
    args = p.parse_args()

    if args.ctx_only:
        build_ctx_index_for_existing(args.ctx_only)
    else:
        if not all([args.frames_dir, args.masks_dir, args.output_dir]):
            p.error("--frames_dir, --masks_dir, and --output_dir are required unless --ctx-only is set")
        generate(args.frames_dir, args.masks_dir, args.output_dir,
                 args.val_frac, args.seed, audio_root=args.audio_root)


if __name__ == "__main__":
    main()
