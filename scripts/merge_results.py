"""Merge per-shard CSVs produced by parallel seggpt_touch_inference runs.

Usage:
    python scripts/merge_results.py results/epic_kitchen_val_seggpt_touch_results_*.csv
    python scripts/merge_results.py results/epic_kitchen_val_seggpt_touch_results_*.csv --output results/epic_kitchen_val_seggpt_touch_results.csv
"""

import argparse
import csv
import glob
import sys
from pathlib import Path

_CSV_FIELDS = [
    "frame_id", "video_id", "frame_path", "audio_path",
    "label", "prediction", "x_touch", "y_touch",
]


def _print_metrics(results: list[dict]) -> None:
    labels = [int(r["label"]) for r in results]
    preds  = [int(r["prediction"]) for r in results]
    tp = sum(l == 1 and p == 1 for l, p in zip(labels, preds))
    tn = sum(l == 0 and p == 0 for l, p in zip(labels, preds))
    fp = sum(l == 0 and p == 1 for l, p in zip(labels, preds))
    fn = sum(l == 1 and p == 0 for l, p in zip(labels, preds))
    n  = len(results)
    acc  = (tp + tn) / n
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    sep  = "─" * 40
    print(f"\n{sep}")
    print(f"  Samples   : {n}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "shards", nargs="+", metavar="SHARD_CSV",
        help="Shard CSV files to merge (glob patterns are expanded by the shell).",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output CSV path. Defaults to the shard path with the trailing _<N> removed.",
    )
    args = parser.parse_args()

    # Expand any un-expanded globs (e.g. when called from Python directly).
    paths: list[Path] = []
    for s in args.shards:
        expanded = sorted(glob.glob(s))
        paths.extend(Path(p) for p in (expanded or [s]))

    missing = [p for p in paths if not p.exists()]
    if missing:
        for p in missing:
            print(f"ERROR: file not found: {p}", file=sys.stderr)
        sys.exit(1)

    rows: list[dict] = []
    for path in sorted(paths):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            shard_rows = list(reader)
        print(f"  {path.name}: {len(shard_rows)} rows")
        rows.extend(shard_rows)

    if not rows:
        print("No rows found across shards.", file=sys.stderr)
        sys.exit(1)

    if args.output:
        out_path = args.output
    else:
        # Infer merged path by stripping the trailing _<digits> from the first shard name.
        stem = paths[0].stem
        import re
        stem = re.sub(r"_\d+$", "", stem)
        out_path = paths[0].parent / f"{stem}.csv"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nMerged {len(rows)} rows from {len(paths)} shards → {out_path}")
    _print_metrics(rows)


if __name__ == "__main__":
    main()
