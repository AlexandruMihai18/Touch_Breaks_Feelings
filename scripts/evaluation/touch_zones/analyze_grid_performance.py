"""Analyze model touch-detection performance per spatial grid zone.

Reads a CSV produced by a model evaluation run with columns:
    frame_id, video_id, frame_path, audio_path, label, prediction, x_touch, y_touch

For each grid zone (x_touch, y_touch), computes how accurately the model predicted
touch for samples whose touch event fell in that zone, then renders an N×N heatmap.

Note on metrics
---------------
x_touch / y_touch are only set for touch-positive samples.  The per-zone metric
is therefore computed exclusively over positive samples in that zone, which makes
it equivalent to per-zone *recall* (hit rate).  Global accuracy and F1 are printed
as summary statistics and use all rows in the CSV.

Usage
-----
    python scripts/analyze_grid_performance.py results/predictions.csv \\
        [--metric recall|accuracy] [--grid 8] [--output heatmap.png]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import enrich_df


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def compute_zone_recall(df: pd.DataFrame, grid: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (recall_grid, count_grid) arrays shaped (grid, grid).

    recall_grid[y, x] = fraction of touch samples in zone (x, y) predicted correctly.
    NaN where no samples exist.
    """
    recall = np.full((grid, grid), np.nan)
    count  = np.zeros((grid, grid), dtype=int)

    touch = df.dropna(subset=["x_touch", "y_touch"]).copy()
    touch["x_touch"] = touch["x_touch"].astype(int)
    touch["y_touch"] = touch["y_touch"].astype(int)
    touch = touch[(touch["x_touch"].between(0, grid - 1)) & (touch["y_touch"].between(0, grid - 1))]

    for (y, x), group in touch.groupby(["y_touch", "x_touch"]):
        n = len(group)
        count[y, x] = n
        recall[y, x] = (group["prediction"] == group["label"]).sum() / n

    return recall, count


def global_stats(df: pd.DataFrame) -> dict:
    y_true = df["label"].astype(int)
    y_pred = df["prediction"].astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1":       f1_score(y_true, y_pred, zero_division=0),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "total": len(df),
    }


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

def plot_heatmap(
    recall: np.ndarray,
    count: np.ndarray,
    metric_label: str,
    output_path: Path,
) -> None:
    grid = recall.shape[0]

    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="#dddddd")

    fig, ax = plt.subplots(figsize=(7, 6))
    masked = np.ma.masked_invalid(recall)
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="equal")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric_label, fontsize=11)

    for y in range(grid):
        for x in range(grid):
            n = count[y, x]
            if n == 0:
                ax.text(x, y, "–", ha="center", va="center", fontsize=9, color="#999999")
            else:
                v = recall[y, x]
                cell_txt = f"{v:.2f}\nn={n}"
                text_color = "white" if v < 0.35 or v > 0.75 else "black"
                ax.text(x, y, cell_txt, ha="center", va="center", fontsize=7.5, color=text_color)

    ax.set_xticks(range(grid))
    ax.set_yticks(range(grid))
    ax.set_xticklabels([str(i) for i in range(grid)])
    ax.set_yticklabels([str(i) for i in range(grid)])
    ax.set_xlabel("x zone  (left → right)", fontsize=11)
    ax.set_ylabel("y zone  (top → bottom)", fontsize=11)
    ax.set_title(f"Per-zone {metric_label}\n(touch-positive samples only)", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path,
                        help="CSV with frame_id,video_id,...,label,prediction,x_touch,y_touch")
    parser.add_argument("--metric", choices=["recall", "accuracy"], default="recall",
                        help="Per-zone metric to display (default: recall). "
                             "Both are equivalent for touch-only samples; choose the label you prefer.")
    parser.add_argument("--grid", type=int, default=8, metavar="N",
                        help="Grid dimension — must match what was used in annotate_touch_zones.py (default: 8)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output PNG path (default: <csv_stem>_heatmap.png next to the CSV)")
    parser.add_argument("--annotations", nargs="+", type=Path, default=None,
                        help="Annotation JSON file(s) to join x_touch/y_touch into the CSV")
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(args.csv)

    df = pd.read_csv(args.csv)
    if args.annotations:
        df = enrich_df(df, args.annotations)

    required = {"label", "prediction", "x_touch", "y_touch"}
    if missing := required - set(df.columns):
        raise ValueError(f"CSV is missing columns: {missing}")

    output = args.output or args.csv.with_name(args.csv.stem + "_heatmap.png")

    recall, count = compute_zone_recall(df, args.grid)
    metric_label = "Recall (hit rate)" if args.metric == "recall" else "Accuracy"
    plot_heatmap(recall, count, metric_label, output)
    print(f"Heatmap saved → {output}")

    stats = global_stats(df)
    print(
        f"\nGlobal stats  (n={stats['total']})\n"
        f"  accuracy : {stats['accuracy']:.3f}\n"
        f"  F1       : {stats['f1']:.3f}\n"
        f"  TP={stats['tp']}  FP={stats['fp']}  FN={stats['fn']}  TN={stats['tn']}"
    )

    # Per-zone summary: zones sorted by recall (descending)
    rows = []
    for y in range(args.grid):
        for x in range(args.grid):
            n = count[y, x]
            if n > 0:
                rows.append({"x": x, "y": y, "n": n, "recall": recall[y, x]})
    if rows:
        summary = pd.DataFrame(rows).sort_values("recall", ascending=False)
        print(f"\nPer-zone recall (top 5):\n{summary.head().to_string(index=False)}")
        print(f"\nPer-zone recall (bottom 5):\n{summary.tail().to_string(index=False)}")


if __name__ == "__main__":
    main()
