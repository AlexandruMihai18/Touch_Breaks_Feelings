"""
Build Greatest Hits context-index files.

Reads train.json / val.json and writes:

  annotations/train_ctx_index.json   — { video_id: [sample_indices] }
  annotations/val_ctx_index.json     — { video_id: [sample_indices] }

The context index maps each video_id to the list of sample indices from that
video, used by TouchPairDataset to sample context frames from the same video
as the query.

Usage
-----
    python scripts/greatest_hits/generate_ctx_index.py

    python scripts/greatest_hits/generate_ctx_index.py \\
        --annotations-dir /scratch-shared/<user>/greatest_hits/annotations
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from inference_script.shared import GH_MASKS_ROOT

_ANNOTATIONS_DIR = GH_MASKS_ROOT.parent / "annotations"


def build_ctx_index(entries: list[dict]) -> dict[str, list[int]]:
    index: dict[str, list[int]] = defaultdict(list)
    for idx, s in enumerate(entries):
        video_id = s.get("video_id", "")
        if video_id:
            index[video_id].append(idx)
    return dict(index)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build ctx_index JSONs (video_id → sample indices) for train and val splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--annotations-dir",
        type=Path,
        default=_ANNOTATIONS_DIR,
        help="Directory containing train.json / val.json.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  Greatest Hits — ctx_index generation")
    print(f"  annotations dir : {args.annotations_dir}")
    print("=" * 60)

    for split in ("train", "val"):
        json_path = args.annotations_dir / f"{split}.json"
        if not json_path.exists():
            print(f"  {split}.json not found — skipping.")
            continue

        entries = json.loads(json_path.read_text())
        ctx_index = build_ctx_index(entries)

        total = sum(len(v) for v in ctx_index.values())
        print(f"\n  {split}: {len(entries):,} samples, {len(ctx_index)} videos, {total:,} indexed")
        for vid, idxs in ctx_index.items():
            print(f"    {vid}: {len(idxs)} frames")

        out_path = args.annotations_dir / f"{split}_ctx_index.json"
        out_path.write_text(json.dumps(ctx_index, indent=2))
        print(f"  → written {out_path.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
