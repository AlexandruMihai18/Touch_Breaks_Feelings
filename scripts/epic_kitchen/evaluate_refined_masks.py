"""
Evaluate depth-refined touch masks as binary classification.

Ground truth comes from the annotation JSON files (train.json / val.json):
  positive (touch)    — entries where type == "touch"
  negative (no-touch) — entries where type == "no-touch"

Predictions are derived from the *_touch_refined.png masks written by
refine_touch_masks.py:
  "touch"    — refined mask contains at least one positive pixel
  "no-touch" — refined mask is all zeros

Reports:
  • Console: confusion matrix + Accuracy / Precision / Recall / F1
             per split and overall combined
  • Per-object-category table (console + optional CSV)
    — objects with < --min-samples entries are grouped into "(other)"

Usage:
    python scripts/epic_kitchen/evaluate_refined_masks.py
    python scripts/epic_kitchen/evaluate_refined_masks.py --splits train val
    python scripts/epic_kitchen/evaluate_refined_masks.py \\
        --splits train val \\
        --output results/ek_refined_eval.csv \\
        --min-samples 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

_ANNO_DIR = _PROJECT_ROOT / "data" / "epic_kitchen" / "annotations"
_OTHER = "(other)"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _refined_path(touch_path: str) -> Path:
    p = Path(touch_path)
    return p.parent / (p.stem + "_refined" + p.suffix)


def load_entries(anno_dir: Path, splits: list[str]) -> pd.DataFrame:
    records = []
    for split in splits:
        path = anno_dir / f"{split}.json"
        if not path.exists():
            print(f"  Warning: {path} not found — skipping")
            continue
        entries = json.loads(path.read_text())
        for e in entries:
            touch_path = e.get("target_path") or e.get("touch_mask_path", "")
            records.append({
                "split": split,
                "video_id": e.get("video_id", ""),
                "object_name": e.get("object_name", "(unknown)"),
                "gt": e["type"],
                "touch_path": touch_path,
                "refined_path": str(_refined_path(touch_path)) if touch_path else "",
            })
    return pd.DataFrame(records)


def predict_from_masks(df: pd.DataFrame) -> pd.DataFrame:
    """Read each refined mask and set pred = 'touch' / 'no-touch' / 'missing'."""
    preds = []
    missing = 0
    for row in df.itertuples():
        rp = Path(row.refined_path)
        if not rp.exists():
            missing += 1
            preds.append("missing")
            continue
        arr = np.array(Image.open(rp).convert("L"))
        preds.append("touch" if np.any(arr > 0) else "no-touch")
    if missing:
        print(f"  Warning: {missing} refined mask(s) not found — excluded from metrics")
    df = df.copy()
    df["pred"] = preds
    return df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _safe_div(num: int, denom: int) -> float:
    return num / denom if denom > 0 else float("nan")


def compute_metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    precision = _safe_div(tp, tp + fp)
    recall    = _safe_div(tp, tp + fn)
    f1        = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else float("nan")
    accuracy  = _safe_div(tp + tn, tp + fp + fn + tn)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                precision=precision, recall=recall, f1=f1, accuracy=accuracy)


def confusion_counts(sub: pd.DataFrame) -> tuple[int, int, int, int]:
    tp = int(((sub["gt"] == "touch")    & (sub["pred"] == "touch")).sum())
    fp = int(((sub["gt"] == "no-touch") & (sub["pred"] == "touch")).sum())
    fn = int(((sub["gt"] == "touch")    & (sub["pred"] == "no-touch")).sum())
    tn = int(((sub["gt"] == "no-touch") & (sub["pred"] == "no-touch")).sum())
    return tp, fp, fn, tn


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(val: float, pct: bool = True) -> str:
    if val != val:
        return "   N/A"
    return f"{val * 100:6.1f}%" if pct else f"{val:6.3f}"


def _print_metrics_row(label: str, m: dict, width: int = 30) -> None:
    print(
        f"  {label:<{width}}  "
        f"TP={m['tp']:4d}  FP={m['fp']:4d}  FN={m['fn']:4d}  TN={m['tn']:4d}  "
        f"Acc={_fmt(m['accuracy'])}  "
        f"P={_fmt(m['precision'])}  "
        f"R={_fmt(m['recall'])}  "
        f"F1={_fmt(m['f1'])}"
    )


def print_section(title: str, sub: pd.DataFrame) -> dict:
    tp, fp, fn, tn = confusion_counts(sub)
    m = compute_metrics(tp, fp, fn, tn)
    n = len(sub)
    print(f"\n── {title} ({n} frames) {'─' * max(0, 70 - len(title) - len(str(n)))}")
    print(f"  Confusion matrix (rows=GT, cols=pred):")
    print(f"                    pred=touch  pred=no-touch")
    print(f"    GT=touch           {tp:6d}         {fn:6d}")
    print(f"    GT=no-touch        {fp:6d}         {tn:6d}")
    print()
    _print_metrics_row(title, m)
    return m


def print_per_object(df: pd.DataFrame, min_samples: int) -> list[dict]:
    print(f"\n── Per object category (min_samples={min_samples}) {'─' * 40}")
    rows = []
    grouped = df.groupby("object_name", sort=True)
    for obj, sub in grouped:
        if len(sub) < min_samples:
            obj_label = _OTHER
        else:
            obj_label = str(obj)
        tp, fp, fn, tn = confusion_counts(sub)
        m = compute_metrics(tp, fp, fn, tn)
        rows.append({"object_name": obj_label, "n": len(sub), **m})

    # Merge "(other)" rows
    result: dict[str, dict] = {}
    for r in rows:
        key = r["object_name"]
        if key not in result:
            result[key] = r.copy()
        else:
            prev = result[key]
            tp2 = prev["tp"] + r["tp"]
            fp2 = prev["fp"] + r["fp"]
            fn2 = prev["fn"] + r["fn"]
            tn2 = prev["tn"] + r["tn"]
            n2  = prev["n"]  + r["n"]
            result[key] = {"object_name": key, "n": n2,
                           **compute_metrics(tp2, fp2, fn2, tn2)}

    per_obj = sorted(result.values(), key=lambda r: -r["n"])
    for r in per_obj:
        _print_metrics_row(f"{r['object_name']} (n={r['n']})", r, width=34)

    return per_obj


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate depth-refined touch masks as binary classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--anno-dir", type=Path, default=_ANNO_DIR,
        help="Directory containing train.json / val.json annotation files.",
    )
    p.add_argument(
        "--splits", nargs="+", default=["train", "val"], metavar="SPLIT",
        help="Which annotation splits to evaluate.",
    )
    p.add_argument(
        "--min-samples", type=int, default=5,
        help="Object categories with fewer entries are merged into '(other)'.",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Optional path to write per-frame results as CSV.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 90)
    print("  Epic Kitchen refined touch masks — evaluation report")
    print("=" * 90)

    df = load_entries(args.anno_dir, args.splits)
    if df.empty:
        print("No entries found.")
        sys.exit(1)

    print(f"  Entries loaded: {len(df):,}  ({', '.join(f'{s}={len(df[df.split==s])}' for s in args.splits)})")
    print("  Reading refined masks…")
    df = predict_from_masks(df)
    df = df[df["pred"] != "missing"].copy()

    # ── Per split ────────────────────────────────────────────────────────────
    for split in args.splits:
        sub = df[df["split"] == split]
        if not sub.empty:
            print_section(split, sub)

    # ── Combined ─────────────────────────────────────────────────────────────
    if len(args.splits) > 1:
        print_section("combined (train + val)", df)

    # ── Per object ───────────────────────────────────────────────────────────
    per_obj = print_per_object(df, args.min_samples)

    print("\n" + "=" * 90)

    # ── CSV output ───────────────────────────────────────────────────────────
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"\nPer-frame results      → {args.output}")

        obj_path = args.output.with_name(args.output.stem + "_per_object.csv")
        pd.DataFrame(per_obj).to_csv(obj_path, index=False)
        print(f"Per-object summary     → {obj_path}")


if __name__ == "__main__":
    main()
