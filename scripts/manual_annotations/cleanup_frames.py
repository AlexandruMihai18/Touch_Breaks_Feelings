"""
One-time cleanup: keep the N most-recently-modified frames per video folder and
remove everything else — frame files, all five associated mask files, and the
corresponding entries in manifest.json / dataset.json.

The global annotation splits (annotations/train.json, annotations/val.json) are
deleted because they become stale; re-run generate_annotations.py afterwards.

Always do a --dry-run first to verify what will be deleted.

Usage
-----
    # Preview
    python scripts/greatest_hits/cleanup_frames.py --dry-run

    # Execute
    python scripts/greatest_hits/cleanup_frames.py --keep 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FRAMES_ROOT = Path("data/manual_annotations/frames")
MASKS_ROOT = Path("data/manual_annotations/masks")
ANNOTATIONS_ROOT = Path("data/manual_annotations/annotations")

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# Suffixes saved by annotate.py for each frame stem
_MASK_SUFFIXES = ("_touch.png", "_object.png", "_hand.png", "_depth.png", ".jpg")


def _frame_files(folder: Path) -> list[Path]:
    return [p for p in sorted(folder.iterdir()) if p.suffix.lower() in IMAGE_EXTS]


def _del(path: Path, dry_run: bool) -> None:
    print(f"    rm  {path}")
    if not dry_run:
        path.unlink()


def cleanup(keep: int = 20, dry_run: bool = False) -> None:
    if not FRAMES_ROOT.exists():
        raise FileNotFoundError(f"Frames root not found: {FRAMES_ROOT}")

    for video_dir in sorted(FRAMES_ROOT.iterdir()):
        if not video_dir.is_dir():
            continue

        frames = _frame_files(video_dir)
        print(
            f"\n{'[DRY] ' if dry_run else ''}{video_dir.name}: {len(frames)} frames found"
        )

        if len(frames) <= keep:
            print(f"  → nothing to remove (≤ {keep})")
            continue

        # Sort by mtime descending so the most recently written frames come first
        by_mtime = sorted(frames, key=lambda p: p.stat().st_mtime, reverse=True)
        keep_stems = {p.stem for p in by_mtime[:keep]}
        drop_stems = {p.stem for p in by_mtime[keep:]}
        print(f"  keeping {len(keep_stems)}, dropping {len(drop_stems)}")

        # Drop source frames
        for p in frames:
            if p.stem in drop_stems:
                _del(p, dry_run)

        # Drop mask files
        mask_dir = MASKS_ROOT / video_dir.name
        if mask_dir.exists():
            for stem in sorted(drop_stems):
                for suffix in _MASK_SUFFIXES:
                    mp = mask_dir / f"{stem}{suffix}"
                    if mp.exists():
                        _del(mp, dry_run)
                    else:
                        print(f"    skip {mp.name} (not found)")

            # Update manifest.json
            manifest_path = mask_dir / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = json.load(f)
                old_n = len(manifest.get("frames", []))
                manifest["frames"] = [
                    fr
                    for fr in manifest["frames"]
                    if Path(fr["image_path"]).stem in keep_stems
                ]
                print(f"  manifest.json: {old_n} → {len(manifest['frames'])} entries")
                if not dry_run:
                    with open(manifest_path, "w") as f:
                        json.dump(manifest, f, indent=2)

            # Update dataset.json
            dataset_path = mask_dir / "dataset.json"
            if dataset_path.exists():
                with open(dataset_path) as f:
                    dataset = json.load(f)
                old_n = len(dataset)
                dataset = [
                    row for row in dataset if Path(row["image_path"]).stem in keep_stems
                ]
                print(f"  dataset.json:  {old_n} → {len(dataset)} rows")
                if not dry_run:
                    with open(dataset_path, "w") as f:
                        json.dump(dataset, f, indent=2)
        else:
            print(f"  mask dir not found: {mask_dir}")

    # Invalidate global splits — they reference absolute paths that are now stale
    print()
    for split_file in ("train.json", "val.json"):
        p = ANNOTATIONS_ROOT / split_file
        if p.exists():
            print(f"Removing stale split: {p}")
            if not dry_run:
                p.unlink()

    if dry_run:
        print("\n[dry-run] No files were modified. Re-run without --dry-run to apply.")
    else:
        print("\nDone. Re-run generate_annotations.py to rebuild train/val splits.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Keep the N most-recently-modified frames per video folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=20,
        help="Number of frames to keep per folder (sorted by mtime, newest first)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be deleted without touching any files",
    )
    args = parser.parse_args()
    cleanup(keep=args.keep, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
