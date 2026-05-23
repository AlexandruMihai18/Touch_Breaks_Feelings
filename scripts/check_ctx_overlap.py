"""
Report train/val class overlap for SegGPT context selection.

For each dataset, loads train_ctx_index.json and val_ctx_index.json and prints:
  - How many val classes have ≥1 train sample  → will use train context (ideal)
  - How many val classes have 0 train samples  → fall back to val
      ∟ of those, how many have >1 val sample  → val fallback OK
      ∟ of those, how many have exactly 1 val  → will be SKIPPED
  - Size distributions for both train and val pools per class

Usage:
  python scripts/check_ctx_overlap.py --data-root /scratch-shared/<user>
"""

import argparse
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

_DATASETS = {
    "epic_kitchen": "object_name",
    "greatest_hits": "video_id",
}


def _load_index(path: Path) -> dict[str, list[int]]:
    with open(path) as f:
        return json.load(f)


def analyze(dataset: str, data_root: Path) -> None:
    anno_dir = data_root / dataset / "annotations"
    val_idx   = _load_index(anno_dir / "val_ctx_index.json")
    train_idx = _load_index(anno_dir / "train_ctx_index.json")

    val_classes = set(val_idx)
    train_classes = set(train_idx)

    has_train   = sorted(c for c in val_classes if c in train_classes)
    no_train    = sorted(c for c in val_classes if c not in train_classes)
    fallback_ok = [c for c in no_train if len(val_idx[c]) > 1]
    will_skip   = [c for c in no_train if len(val_idx[c]) <= 1]

    total = len(val_classes)
    sep = "─" * 52

    print(f"\n{'═' * 52}")
    print(f"  {dataset}")
    print(f"{'═' * 52}")
    print(f"  Val classes total          : {total}")
    print(f"  {sep}")
    print(f"  Train context available    : {len(has_train):>4}  ({100*len(has_train)/total:.1f}%)")
    print(f"  No train context           : {len(no_train):>4}  ({100*len(no_train)/total:.1f}%)")
    print(f"    ↳ val fallback (>1 val)  : {len(fallback_ok):>4}")
    print(f"    ↳ will be SKIPPED (≤1)  : {len(will_skip):>4}")

    if has_train:
        train_sizes = [len(train_idx[c]) for c in has_train]
        val_sizes   = [len(val_idx[c])   for c in has_train]
        print(f"\n  [classes with train context]")
        print(f"  train samples/class  min={min(train_sizes)}  max={max(train_sizes)}  avg={sum(train_sizes)/len(train_sizes):.1f}")
        print(f"  val   queries/class  min={min(val_sizes)}  max={max(val_sizes)}  avg={sum(val_sizes)/len(val_sizes):.1f}")

    if fallback_ok:
        fb_val_sizes = [len(val_idx[c]) for c in fallback_ok]
        print(f"\n  [val-fallback classes (>{1} val sample)]")
        print(f"  val samples/class  min={min(fb_val_sizes)}  max={max(fb_val_sizes)}  avg={sum(fb_val_sizes)/len(fb_val_sizes):.1f}")

    if will_skip:
        print(f"\n  [classes that will be SKIPPED]")
        for c in will_skip:
            print(f"    {c!r}  (val samples: {len(val_idx[c])})")

    # Classes in train but not in val — informational
    train_only = train_classes - val_classes
    print(f"\n  Train-only classes (not in val): {len(train_only)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-root", type=Path, default=_ROOT / "data",
        help="Root under which dataset folders live (same as inference script).",
    )
    parser.add_argument(
        "datasets", nargs="*", default=list(_DATASETS),
        choices=list(_DATASETS), metavar="DATASET",
        help="Datasets to analyse (default: all).",
    )
    args = parser.parse_args()

    for dataset in args.datasets:
        try:
            analyze(dataset, args.data_root)
        except FileNotFoundError as exc:
            print(f"\n[{dataset}] missing file — {exc}")


if __name__ == "__main__":
    main()
