"""
Merge two or more annotation JSON files into a single JSON (no re-splitting).

Deduplication key is ``target_path`` (encodes frame stem + pair_idx, unique
per annotation entry).  Entries from later --inputs files are dropped when
their key already appeared in an earlier file.

Usage
-----
    # Collapse the Phase-5 train/val split back into one train.json
    python -m scripts.epic_kitchen.merge_annotations \
        --inputs data/epic_kitchen/annotations/train.json \
                 data/epic_kitchen/annotations/val.json \
        --output data/epic_kitchen/annotations/train.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def merge(inputs: list[Path], output: Path) -> None:
    all_pairs: list[dict] = []
    seen: set[str] = set()

    for inp in inputs:
        raw = json.loads(inp.read_text())
        before = len(all_pairs)
        for entry in raw:
            key = entry.get("target_path", "")
            if key and key not in seen:
                seen.add(key)
                all_pairs.append(entry)
        added = len(all_pairs) - before
        print(f"  {inp.name}: {len(raw)} entries, {added} kept, {len(raw)-added} duplicates dropped")

    n_touch  = sum(1 for p in all_pairs if p.get("type") == "touch")
    n_videos = len({p["video_id"] for p in all_pairs})
    print(f"\nMerged: {len(all_pairs):,} pairs ({n_touch:,} touch, {len(all_pairs)-n_touch:,} no-touch) across {n_videos} videos")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(all_pairs, indent=2))
    print(f"Written → {output}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Merge annotation JSONs into a single file (no re-splitting)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--inputs", nargs="+", required=True,
                   help="Annotation JSON files to merge")
    p.add_argument("--output", required=True,
                   help="Output JSON file path")
    args = p.parse_args()

    merge(
        inputs=[Path(i) for i in args.inputs],
        output=Path(args.output),
    )


if __name__ == "__main__":
    main()
