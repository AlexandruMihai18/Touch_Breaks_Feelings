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
import matplotlib as mpl
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import enrich_df
import plot_style


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

    touch = df.dropna(subset=["x_touch_annot", "y_touch_annot"]).copy()
    touch["x_touch_annot"] = touch["x_touch_annot"].astype(int)
    touch["y_touch_annot"] = touch["y_touch_annot"].astype(int)
    touch = touch[(touch["x_touch_annot"].between(0, grid - 1)) & (touch["y_touch_annot"].between(0, grid - 1))]

    for (y, x), group in touch.groupby(["y_touch_annot", "x_touch_annot"]):
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
    plot_style.apply()

    grid = recall.shape[0]

    # Use RdYlGn but mask invalid cells with a light neutral grey
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="#EBEBEB")

    fig, ax = plt.subplots(figsize=(6.0, 5.5))

    valid = recall[~np.isnan(recall) & (recall > 0)]
    vmin = float(valid.min()) if valid.size > 0 else 0.0

    masked = np.ma.masked_invalid(recall)
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=1, aspect="equal")

    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046, aspect=20)
    cbar.set_label(metric_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_linewidth(0.5)

    for y in range(grid):
        for x in range(grid):
            n = count[y, x]
            if n == 0:
                ax.text(x, y, "–", ha="center", va="center",
                        fontsize=8, color="#AAAAAA")
            else:
                v = recall[y, x]
                txt = f"{v:.2f}\n({n})"
                fg = "white" if (v < vmin + (1 - vmin) * 0.25
                                 or v > vmin + (1 - vmin) * 0.78) else plot_style.DARK
                ax.text(x, y, txt, ha="center", va="center",
                        fontsize=7.0, color=fg, linespacing=1.35)

    # Grid lines between cells
    for i in range(grid + 1):
        ax.axhline(i - 0.5, color="white", linewidth=0.8)
        ax.axvline(i - 0.5, color="white", linewidth=0.8)

    ax.set_xticks(range(grid))
    ax.set_yticks(range(grid))
    ax.set_xticklabels([str(i) for i in range(grid)], fontsize=8)
    ax.set_yticklabels([str(i) for i in range(grid)], fontsize=8)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("x zone  (left → right)", labelpad=6)
    ax.set_ylabel("y zone  (bottom → top)", labelpad=6)
    ax.set_title(f"Per-zone {metric_label}\n(touch-positive samples only)", pad=14)

    # Turn off the y-axis grid that rcParams enables (grid on imshow looks bad)
    ax.grid(False)

    plt.savefig(output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path,
                        help="CSV with frame_id,video_id,...,label,prediction,x_touch,y_touch")
    _METRICS = ["recall", "accuracy"]
    parser.add_argument("--metric", choices=_METRICS, default="recall",
                        help="Metric label for the heatmap (default: recall). "
                             "x_touch/y_touch are only annotated for touch-positive samples, so "
                             "precision and F1 are not computable per zone — recall and accuracy "
                             "are equivalent here.")
    parser.add_argument("--all-metrics", action="store_true",
                        help=f"Generate one figure per metric {_METRICS}, each suffixed with the metric name. "
                             "Overrides --metric.")
    parser.add_argument("--grid", type=int, default=8, metavar="N",
                        help="Grid dimension — must match what was used in annotate_touch_zones.py (default: 8)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output PNG path (default: <csv_stem>_heatmap.png next to the CSV)")
    parser.add_argument("--annotations", nargs="+", type=Path, required=True,
                        help="Annotation JSON file(s) providing ground-truth grid zones (required; not in prediction CSV)")
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(args.csv)

    df = pd.read_csv(args.csv)
    df = enrich_df(df, args.annotations)
    df = df.rename(columns={"x_touch": "x_touch_annot", "y_touch": "y_touch_annot"})

    required = {"label", "prediction", "x_touch_annot", "y_touch_annot"}
    if missing := required - set(df.columns):
        raise ValueError(f"CSV is missing columns: {missing}")

    base_output = args.output or args.csv.with_name(args.csv.stem + "_heatmap.png")
    recall, count = compute_zone_recall(df, args.grid)

    metrics = _METRICS if args.all_metrics else [args.metric]
    for metric in metrics:
        metric_label = "Recall (hit rate)" if metric == "recall" else "Accuracy"
        out = plot_style.with_metric_suffix(base_output, metric) if args.all_metrics else base_output
        plot_heatmap(recall, count, metric_label, out)
        print(f"Heatmap saved → {out}")

    stats = global_stats(df)
    print(
        f"\nGlobal stats  (n={stats['total']})\n"
        f"  accuracy : {stats['accuracy']:.3f}\n"
        f"  F1       : {stats['f1']:.3f}\n"
        f"  Precision: {stats['tp'] / (stats['tp'] + stats['fp']):.3f}\n"
        f"  Recall   : {stats['tp'] / (stats['tp'] + stats['fn']):.3f}\n"
        f"  TP={stats['tp']}  FP={stats['fp']}  FN={stats['fn']}  TN={stats['tn']}"
    )

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
