"""
Evaluate the Greatest Hits auto-annotation pipeline as binary classification.

Ground-truth labels come from train.json + val.json (built by 3_generate_gh_gt_annotations.py):
  positive (touch)    — entries where type is "touch"
  negative (no-touch) — entries where type is "no-touch"

Predictions come from per-video dataset.json files written by the annotation pipeline:
  "touch"    — pipeline found touch pixels
  "no-touch" — pipeline ran but found no touch pixels

Train/val splits are read from the official Greatest Hits split txt files.  Each line
in those files is a bare timestamp ID (e.g. 2015-02-16-16-49-06); the script appends
"_denoised" to match the video_id format used in the annotation JSONs.  Videos that
appear in neither split file are labelled "unassigned" and included in overall metrics only.

Frames in GT with no matching dataset.json entry are excluded from metrics
but tracked as coverage gaps and reported in the summary.

Reports:
  • Console: per-split confusion matrix + metrics, overall, per-material breakdown
  • CSV (optional, --output): per-frame result table (includes split column)

Usage:
    python scripts/greatest_hits/evaluate_annotation_pipeline.py

    python scripts/greatest_hits/evaluate_annotation_pipeline.py \\
        --masks-dir        data/greatest_hits/masks \\
        --annotations-dir  data/greatest_hits/annotations \\
        --train-split      data/greatest_hits/vis-data-256/vis-data-256/train.txt \\
        --test-split       data/greatest_hits/vis-data-256/vis-data-256/test.txt \\
        --output           results/gh_eval.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2]))

from inference_script.shared import GH_ANNO_DIR, GH_MASKS_ROOT

_UNLABELED = "(unlabeled)"
_NO_ACTION = "(no-touch)"
_UNSPECIFIED_ACTION = "(unspecified)"
_GH_RAW_DIR = GH_MASKS_ROOT.parent / "vis-data-256" / "vis-data-256"


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_ground_truth(annotations_dir: Path) -> pd.DataFrame:
    """Load train.json + val.json and assign binary GT labels."""
    import json as _json
    records = []
    for split_name in ("train", "val"):
        path = annotations_dir / f"{split_name}.json"
        if not path.exists():
            continue
        for entry in _json.loads(path.read_text()):
            is_touch = entry["type"] == "touch"
            records.append({
                "video_id":       entry["video_id"],
                "frame_idx":      entry["frame_idx"],
                "gt":             entry["type"],
                "material_group": entry["material"] if entry.get("material") else _UNLABELED,
                "action_group":   (entry.get("action") or _UNSPECIFIED_ACTION) if is_touch else _NO_ACTION,
            })
    return pd.DataFrame(records, columns=["video_id", "frame_idx", "gt", "material_group", "action_group"])


def load_predictions(masks_root: Path) -> pd.DataFrame:
    """Walk masks_root for dataset.json files and collect per-frame predictions."""
    records = []
    for ds_path in sorted(masks_root.rglob("dataset.json")):
        rows = json.loads(ds_path.read_text())
        for row in rows:
            stem = Path(row["image_path"]).stem      # frame_XXXXXX
            frame_idx = int(stem.split("_")[1])
            records.append({
                "video_id": row["video_id"],
                "frame_idx": frame_idx,
                "pred": row["type"],
            })
    return pd.DataFrame(records, columns=["video_id", "frame_idx", "pred"])


def load_split_ids(split_path: Path | None) -> set[str]:
    """Read a split txt file and return video_ids with '_denoised' appended."""
    if split_path is None or not split_path.exists():
        return set()
    ids = set()
    for line in split_path.read_text().splitlines():
        line = line.strip()
        if line:
            ids.add(line + "_denoised")
    return ids


def assign_splits(
    matched: pd.DataFrame,
    train_ids: set[str],
    test_ids: set[str],
) -> pd.DataFrame:
    """Add a 'split' column: 'train', 'test', or 'unassigned'."""
    matched = matched.copy()

    def _label(vid: str) -> str:
        if vid in train_ids:
            return "train"
        if vid in test_ids:
            return "test"
        return "unassigned"

    matched["split"] = matched["video_id"].map(_label)
    return matched


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _safe_div(num: int, denom: int) -> float:
    return num / denom if denom > 0 else float("nan")


def compute_metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    total = tp + fp + fn + tn
    precision = _safe_div(tp, tp + fp)
    recall    = _safe_div(tp, tp + fn)
    f1        = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else float("nan")
    accuracy  = _safe_div(tp + tn, total)
    return dict(
        tp=tp, fp=fp, fn=fn, tn=tn, total=total,
        precision=precision, recall=recall, f1=f1, accuracy=accuracy,
    )


def confusion_counts(df: pd.DataFrame) -> tuple[int, int, int, int]:
    tp = int(((df["gt"] == "touch")    & (df["pred"] == "touch")).sum())
    fp = int(((df["gt"] == "no-touch") & (df["pred"] == "touch")).sum())
    fn = int(((df["gt"] == "touch")    & (df["pred"] == "no-touch")).sum())
    tn = int(((df["gt"] == "no-touch") & (df["pred"] == "no-touch")).sum())
    return tp, fp, fn, tn


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(val: float, pct: bool = False) -> str:
    if val != val:  # NaN
        return "   N/A"
    return f"{val * 100:6.1f}%" if pct else f"{val:6.3f}"


def _print_confusion_matrix(tp: int, fp: int, fn: int, tn: int) -> None:
    print("  Confusion matrix (rows=GT, cols=pred):")
    print(f"                    pred=touch  pred=no-touch")
    print(f"    GT=touch           {tp:6d}         {fn:6d}")
    print(f"    GT=no-touch        {fp:6d}         {tn:6d}")


def _print_metrics_row(label: str, m: dict, width: int = 28) -> None:
    print(
        f"  {label:<{width}}  "
        f"TP={m['tp']:4d}  FP={m['fp']:4d}  FN={m['fn']:4d}  TN={m['tn']:4d}  "
        f"Acc={_fmt(m['accuracy'], pct=True)}  "
        f"P={_fmt(m['precision'], pct=True)}  "
        f"R={_fmt(m['recall'], pct=True)}  "
        f"F1={_fmt(m['f1'], pct=True)}"
    )


def _print_split_section(title: str, sub: pd.DataFrame) -> dict:
    tp, fp, fn, tn = confusion_counts(sub)
    m = compute_metrics(tp, fp, fn, tn)
    n_videos = sub["video_id"].nunique()
    print(f"── {title} ({len(sub):,} frames · {n_videos} videos) {'─' * max(0, 62 - len(title))}")
    _print_confusion_matrix(tp, fp, fn, tn)
    print()
    _print_metrics_row(title, m)
    print()
    return m


def print_report(
    matched: pd.DataFrame,
    gt_total: int,
    pred_total: int,
    has_splits: bool,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Print the full report. Returns (per_split_rows, per_material_rows, per_action_rows)."""
    n_matched = len(matched)
    gt_covered = matched["video_id"].nunique()

    print()
    print("=" * 90)
    print("  Greatest Hits annotation pipeline — evaluation report")
    print("=" * 90)
    print(f"  GT frames (annotations)     : {gt_total:>7,}")
    print(f"  Predicted frames (pipeline) : {pred_total:>7,}")
    print(f"  Matched frames (evaluated)  : {n_matched:>7,}  ({n_matched / gt_total * 100:.1f}% GT coverage)")
    print(f"  Videos with predictions     : {gt_covered:>7,}")
    if has_splits:
        for split_name in ("train", "test", "unassigned"):
            n = int((matched["split"] == split_name).sum())
            if n:
                print(f"    {split_name:<12}: {n:>6,} frames")
    print()

    per_split: list[dict] = []

    # ── Per split ─────────────────────────────────────────────────────────────
    if has_splits:
        for split_name in ("train", "test"):
            sub = matched[matched["split"] == split_name]
            if sub.empty:
                print(f"── {split_name} — no matched frames\n")
                continue
            m = _print_split_section(split_name, sub)
            per_split.append({"split": split_name, **m})

        unassigned = matched[matched["split"] == "unassigned"]
        if not unassigned.empty:
            m = _print_split_section("unassigned", unassigned)
            per_split.append({"split": "unassigned", **m})

    # ── Overall ───────────────────────────────────────────────────────────────
    tp, fp, fn, tn = confusion_counts(matched)
    overall = compute_metrics(tp, fp, fn, tn)
    print("── Overall ─────────────────────────────────────────────────────────────────────")
    _print_confusion_matrix(tp, fp, fn, tn)
    print()
    _print_metrics_row("OVERALL", overall)
    print()

    # ── Per material (across all splits) ──────────────────────────────────────
    print("── Per material ────────────────────────────────────────────────────────────────")
    per_material: list[dict] = []
    for group, sub in matched.groupby("material_group", sort=True):
        stp, sfp, sfn, stn = confusion_counts(sub)
        m = compute_metrics(stp, sfp, sfn, stn)
        _print_metrics_row(str(group), m)
        per_material.append({"material": group, **m})

    # ── Per action (across all splits) ────────────────────────────────────────
    print()
    print("── Per action ──────────────────────────────────────────────────────────────────")
    per_action: list[dict] = []
    if "action_group" in matched.columns:
        for group, sub in matched.groupby("action_group", sort=True):
            stp, sfp, sfn, stn = confusion_counts(sub)
            m = compute_metrics(stp, sfp, sfn, stn)
            _print_metrics_row(str(group), m)
            per_action.append({"action": group, **m})
    else:
        print("  (action data not available — re-run 3_generate_gh_gt_annotations.py)")

    print("=" * 90)
    print()
    return per_split, per_material, per_action


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate Greatest Hits annotation pipeline as binary classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--masks-dir", type=Path, default=GH_MASKS_ROOT,
        help="Root directory containing per-video masks and dataset.json files.",
    )
    p.add_argument(
        "--annotations-dir", type=Path, default=GH_ANNO_DIR,
        help="Directory containing train.json + val.json (built by 3_generate_gh_gt_annotations.py).",
    )
    p.add_argument(
        "--train-split", type=Path, default=_GH_RAW_DIR / "train.txt",
        help="Split txt file listing train video IDs (one bare timestamp per line).",
    )
    p.add_argument(
        "--test-split", type=Path, default=_GH_RAW_DIR / "test.txt",
        help="Split txt file listing test/val video IDs (one bare timestamp per line).",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Optional path to write per-frame results as CSV.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.annotations_dir.exists():
        print(f"Annotations directory not found: {args.annotations_dir}")
        print("Run 3_generate_gh_gt_annotations.py first.")
        sys.exit(1)

    if not args.masks_dir.exists():
        print(f"Masks directory not found: {args.masks_dir}")
        sys.exit(1)

    train_ids = load_split_ids(args.train_split)
    test_ids  = load_split_ids(args.test_split)
    has_splits = bool(train_ids or test_ids)

    if not has_splits:
        print("Warning: no split files found — reporting combined metrics only.")
    else:
        print(f"Splits loaded — train: {len(train_ids)} videos, test: {len(test_ids)} videos")

    gt    = load_ground_truth(args.annotations_dir)
    preds = load_predictions(args.masks_dir)

    if preds.empty:
        print("No dataset.json files found — run annotate_greatest_hits.py first.")
        sys.exit(1)

    matched = gt.merge(preds, on=["video_id", "frame_idx"], how="inner")
    matched = assign_splits(matched, train_ids, test_ids)

    per_split, per_material, per_action = print_report(
        matched,
        gt_total=len(gt),
        pred_total=len(preds),
        has_splits=has_splits,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        matched.to_csv(args.output, index=False)
        print(f"Per-frame results written  → {args.output}")

        if per_split:
            split_path = args.output.with_name(args.output.stem + "_per_split.csv")
            pd.DataFrame(per_split).to_csv(split_path, index=False)
            print(f"Per-split summary written  → {split_path}")

        material_path = args.output.with_name(args.output.stem + "_per_material.csv")
        pd.DataFrame(per_material).to_csv(material_path, index=False)
        print(f"Per-material summary written→ {material_path}")

        if per_action:
            action_path = args.output.with_name(args.output.stem + "_per_action.csv")
            pd.DataFrame(per_action).to_csv(action_path, index=False)
            print(f"Per-action summary written → {action_path}")


if __name__ == "__main__":
    main()
