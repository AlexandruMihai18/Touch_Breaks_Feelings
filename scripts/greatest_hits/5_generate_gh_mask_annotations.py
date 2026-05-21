"""
Enrich Greatest Hits annotation files with predicted mask paths.

Reads train.json / val.json written by generate_gh_gt_annotations.py and adds:
    stick_mask_path, object_mask_path, target_path (touch mask)

Masks are resolved from --masks-dir using each entry's video_id and the frame
stem derived from image_path.  Entries with no masks on disk are kept as-is
so that the annotation files remain complete for evaluation purposes.

Also builds *_ctx_index.json — needed for context sampling at training time.
Only entries with non-empty stick AND object masks are included in the index.

Run this after 4_annotate_greatest_hits.py has completed.

Usage
-----
    python scripts/greatest_hits/generate_gh_mask_annotations.py

    python scripts/greatest_hits/generate_gh_mask_annotations.py \\
        --masks-dir       data/greatest_hits/masks \\
        --annotations-dir data/greatest_hits/annotations
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[2]))
from inference_script.shared import GH_MASKS_ROOT

_ANNOTATIONS_DIR = GH_MASKS_ROOT.parent / "annotations"

_MASK_FIELDS = [
    ("stick_mask_path",  "_stick.png"),
    ("object_mask_path", "_object.png"),
    ("target_path",      "_touch.png"),
]


def _enrich(entry: dict, masks_dir: Path) -> dict:
    video_id = entry["video_id"]
    stem = Path(entry["image_path"]).stem
    mask_dir = masks_dir / video_id

    entry = dict(entry)
    for field, suffix in _MASK_FIELDS:
        p = mask_dir / f"{stem}{suffix}"
        if p.exists():
            entry[field] = str(p)
    return entry


def _valid_context(s: dict) -> bool:
    stick = s.get("stick_mask_path", "")
    obj   = s.get("object_mask_path", "")
    if not stick or not obj:
        return False
    try:
        return (
            bool(np.any(np.array(Image.open(stick).convert("L")) > 0))
            and bool(np.any(np.array(Image.open(obj).convert("L")) > 0))
        )
    except OSError:
        return False


def _context_key(s: dict) -> str:
    material = (s.get("material") or "").strip()
    return material if material else s.get("video_id", "")


def _build_ctx_index(entries: list[dict]) -> dict[str, list[int]]:
    index: dict[str, list[int]] = defaultdict(list)
    for idx, s in enumerate(entries):
        if _valid_context(s):
            index[_context_key(s)].append(idx)
    return dict(index)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Add mask paths to GT annotation files and build ctx_index.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--masks-dir",       type=Path, default=GH_MASKS_ROOT,
                   help="Root directory containing per-video mask folders.")
    p.add_argument("--annotations-dir", type=Path, default=_ANNOTATIONS_DIR,
                   help="Directory containing train.json / val.json to enrich.")
    p.add_argument("--no-ctx-index",    action="store_true",
                   help="Skip building ctx_index files.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  Greatest Hits mask annotation enrichment")
    print(f"  masks dir       : {args.masks_dir}")
    print(f"  annotations dir : {args.annotations_dir}")
    print("=" * 60)

    if not args.masks_dir.exists():
        print(f"Masks directory not found: {args.masks_dir}")
        print("Run 4_annotate_greatest_hits.py first.")
        sys.exit(1)

    for split in ("train", "val"):
        json_path = args.annotations_dir / f"{split}.json"
        if not json_path.exists():
            print(f"  {split}.json not found — skipping "
                  f"(run generate_gh_gt_annotations.py first).")
            continue

        entries  = json.loads(json_path.read_text())
        enriched = [_enrich(e, args.masks_dir) for e in entries]

        n_with_masks = sum(1 for e in enriched if "target_path" in e)
        print(f"\n  {split}: {len(enriched):,} entries, "
              f"{n_with_masks:,} with masks "
              f"({len(enriched) - n_with_masks:,} still pending)")

        json_path.write_text(json.dumps(enriched, indent=2))
        print(f"    → updated {json_path.name}")

        if not args.no_ctx_index:
            print(f"    Building ctx_index…", end=" ", flush=True)
            ctx_index = _build_ctx_index(enriched)
            total_valid = sum(len(v) for v in ctx_index.values())
            print(f"{len(ctx_index)} groups, {total_valid:,} valid entries")
            idx_path = args.annotations_dir / f"{split}_ctx_index.json"
            idx_path.write_text(json.dumps(ctx_index, indent=2))
            print(f"    → written {idx_path.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
