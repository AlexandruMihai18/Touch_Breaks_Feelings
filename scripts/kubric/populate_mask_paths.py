#!/usr/bin/env python3
"""
populate_mask_paths.py — scan existing mask files on disk and backfill a
Kubric annotation JSON with their absolute paths.

For each entry, the script looks for files in:
    {masks_root}/{video_id}/{frame_stem}_{suffix}.png

and maps them to annotation fields:

    _depth.png      → depth_path
    _object1.png    → object1_mask_path
    _object2.png    → object2_mask_path
    _touch_gt.png   → touch_gt_path   (point-of-touch: original simulation mask)
    _touch.png      → target_path     (refined touch: computed from GT depth)

Only files that actually exist on disk are written into the annotation.
Existing non-empty fields are left alone unless --overwrite is given.

The primary success metric reported at the end is: how many touch entries
have touch_gt_path pointing to an existing file after the run.

Usage
-----
    python scripts/kubric/populate_mask_paths.py \\
        --anno-file  /scratch-shared/$USER/kubric_movi_a_256/annotations/val.json \\
        --masks-root /scratch-shared/$USER/kubric/masks \\
        [--overwrite] [--dry-run]
"""

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# (annotation_field, file_suffix)
_FIELD_MAP = [
    ("depth_path",        "_depth.png"),
    ("object1_mask_path", "_object1.png"),
    ("object2_mask_path", "_object2.png"),
    ("touch_gt_path",     "_touch_gt.png"),
    ("target_path",       "_touch.png"),
]


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Backfill Kubric annotation JSON with existing mask file paths.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--anno-file", required=True, type=Path,
        help="Annotation JSON to update (e.g., /scratch-shared/$USER/.../val.json).",
    )
    p.add_argument(
        "--masks-root", required=True, type=Path,
        help="Root directory containing per-video mask subdirectories "
             "(e.g., /scratch-shared/$USER/kubric/masks).",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite fields that already have a non-empty value.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be changed without writing the JSON.",
    )
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _video_id(entry: dict) -> str | None:
    vid = entry.get("video_id")
    if vid is not None:
        return str(vid)
    img = entry.get("image_path") or entry.get("frame_path")
    if img:
        return Path(img).parent.name
    return None


def _frame_stem(entry: dict) -> str | None:
    """Return the bare frame stem, e.g. 'frame_000001', from image_path."""
    img = entry.get("image_path") or entry.get("frame_path")
    if not img:
        return None
    return Path(img).stem


def _is_touch(entry: dict) -> bool:
    return (
        entry.get("type") == "touch"
        or entry.get("sample_type") == "touch"
        or bool(entry.get("target_path"))
        or bool(entry.get("touch_gt_path"))
    )


def _scan(entry: dict, masks_root: Path) -> dict[str, Path]:
    """Return {field: absolute_path} for every file that exists on disk."""
    vid  = _video_id(entry)
    stem = _frame_stem(entry)
    if not vid or not stem:
        return {}

    vid_dir = masks_root / vid
    found: dict[str, Path] = {}
    for field, suffix in _FIELD_MAP:
        candidate = vid_dir / f"{stem}{suffix}"
        if candidate.exists():
            found[field] = candidate
    return found


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if not args.anno_file.exists():
        print(f"Error: annotation file not found: {args.anno_file}")
        sys.exit(1)
    if not args.masks_root.is_dir():
        print(f"Error: masks-root not a directory: {args.masks_root}")
        sys.exit(1)

    entries       = json.loads(args.anno_file.read_text())
    touch_entries = [e for e in entries if _is_touch(e)]

    print("=" * 62)
    print("  Kubric annotation path population")
    print(f"  anno file   : {args.anno_file}")
    print(f"  masks root  : {args.masks_root}")
    print(f"  entries     : {len(entries):,}  ({len(touch_entries):,} touch)")
    print(f"  overwrite   : {args.overwrite}")
    print(f"  dry run     : {args.dry_run}")
    print("=" * 62)

    # Per-field counters: newly_set, already_had, file_missing, no_id
    newly_set:    dict[str, int] = {f: 0 for f, _ in _FIELD_MAP}
    already_had:  dict[str, int] = {f: 0 for f, _ in _FIELD_MAP}
    file_missing: dict[str, int] = {f: 0 for f, _ in _FIELD_MAP}
    no_id = 0

    for entry in entries:
        vid  = _video_id(entry)
        stem = _frame_stem(entry)
        if not vid or not stem:
            no_id += 1
            continue

        found = _scan(entry, args.masks_root)

        for field, _ in _FIELD_MAP:
            if field in found:
                if entry.get(field) and not args.overwrite:
                    already_had[field] += 1
                else:
                    newly_set[field] += 1
                    if not args.dry_run:
                        entry[field] = str(found[field])
            else:
                file_missing[field] += 1

    # ── Final-state touch coverage (reflects actual entry state after run) ────
    touch_with_gt  = sum(
        1 for e in touch_entries
        if e.get("touch_gt_path") and Path(e["touch_gt_path"]).exists()
    )
    touch_with_ref = sum(
        1 for e in touch_entries
        if e.get("target_path") and Path(e["target_path"]).exists()
    )

    # ── Report ────────────────────────────────────────────────────────────────
    print()
    print(f"  {'field':<22}  {'set':>6}  {'kept':>6}  {'missing':>8}")
    print(f"  {'-'*22}  {'-'*6}  {'-'*6}  {'-'*8}")
    for field, _ in _FIELD_MAP:
        print(
            f"  {field:<22}  {newly_set[field]:>6,}  "
            f"{already_had[field]:>6,}  {file_missing[field]:>8,}"
        )

    if no_id:
        print(f"\n  [WARN] {no_id} entries skipped (no video_id / image_path)")

    print()
    print(f"  Touch retrieval summary:")
    print(f"    touch entries total        : {len(touch_entries):,}")
    print(
        f"    with point-of-touch (gt)   : {touch_with_gt:,}"
        f"  ({100 * touch_with_gt / max(len(touch_entries), 1):.1f} %)"
    )
    print(
        f"    with refined touch mask    : {touch_with_ref:,}"
        f"  ({100 * touch_with_ref / max(len(touch_entries), 1):.1f} %)"
    )

    if not args.dry_run:
        args.anno_file.write_text(json.dumps(entries, indent=2))
        print(f"\n  Wrote {args.anno_file}")
    else:
        print("\n  [DRY RUN] No files written.")

    print()
    print("=" * 62)
    print("Done.")


if __name__ == "__main__":
    main()
