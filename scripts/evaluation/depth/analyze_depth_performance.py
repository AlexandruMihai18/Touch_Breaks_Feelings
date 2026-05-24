"""Analyze model touch-detection performance across depth bins.

Reads a prediction CSV with a `depth_touch` column (median grayscale disparity
under the touch mask, populated by annotate_touch_depth.py) and reports recall
for three fixed distance categories.

Disparity scale (Depth-Anything-V2 grayscale, 0–255):
    0–84   → Far     (low disparity = objects far away)
    85–169 → Medium
    170–255 → Close   (high disparity = objects close to camera)

Usage
-----
    python scripts/evaluation/depth/analyze_depth_performance.py \\
        results/predictions.csv \\
        --annotations /path/to/annotations.json \\
        [--output depth_performance.png]
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


# Hard disparity thresholds — low = far, high = near (0–255 grayscale)
_BINS = [
    ("Far",    0,   85),
    ("Medium", 85,  170),
    ("Close",  170, 256),
]

# Sample from inferno to stay consistent with how depth maps are visualised
_BIN_COLORS = [plt.cm.inferno(v) for v in (0.15, 0.50, 0.82)]


def compute_bin_stats(df: pd.DataFrame) -> pd.DataFrame | None:
    touch = df.dropna(subset=["depth_touch"]).copy()
    touch = touch[touch["label"] == 1]
    if len(touch) == 0:
        return None

    rows = []
    for label, lo, hi in _BINS:
        group = touch[(touch["depth_touch"] >= lo) & (touch["depth_touch"] < hi)]
        n = len(group)
        if n == 0:
            continue
        rows.append({
            "bin_label": label,
            "n":         n,
            "recall":    (group["prediction"] == 1).sum() / n,
        })

    return pd.DataFrame(rows) if rows else None


def plot_depth_bars(stats: pd.DataFrame, metric_label: str, output_path: Path) -> None:
    plot_style.apply()

    n = len(stats)
    colors = _BIN_COLORS[:n]

    fig, ax = plt.subplots(figsize=(3.5 + n * 0.8, 4.0))

    ax.bar(
        range(n),
        stats["recall"],
        color=colors,
        edgecolor="white",
        linewidth=0.6,
        width=0.52,
        zorder=3,
    )

    for i, (_, row) in enumerate(stats.iterrows()):
        ax.text(
            i, row["recall"] + 0.025,
            f"{row['recall']:.2f}",
            ha="center", va="bottom", fontsize=11, fontweight="semibold",
            color=plot_style.DARK,
        )

    mean_val = stats["recall"].mean()
    ax.axhline(
        mean_val, color=plot_style.GRAY, linestyle="--", linewidth=1.0,
        label=f"Mean = {mean_val:.2f}", zorder=4,
    )

    ax.set_xticks(range(n))
    ax.set_xticklabels(stats["bin_label"], fontsize=12)
    ax.set_ylabel(metric_label)
    ax.set_ylim(0, 1.15)
    ax.set_xlim(-0.55, n - 0.45)
    ax.set_title("Recall by object distance")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlabel("")

    plt.savefig(output_path)
    plt.close(fig)


def main() -> None:
    _METRICS = ["recall", "accuracy"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--metric", choices=_METRICS, default="recall",
                        help="Metric label for the figure (default: recall). "
                             "depth_touch is only annotated for touch-positive samples, so precision "
                             "and F1 are not computable per bin — recall and accuracy are equivalent here.")
    parser.add_argument("--all-metrics", action="store_true",
                        help=f"Generate one figure per metric {_METRICS}, each suffixed with the metric name. "
                             "Overrides --metric.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--annotations", nargs="+", type=Path, required=True,
                        help="Annotation JSON file(s) providing depth_touch")
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(args.csv)

    df = pd.read_csv(args.csv)
    df = enrich_df(df, args.annotations)

    required = {"label", "prediction", "depth_touch"}
    if missing := required - set(df.columns):
        raise ValueError(f"CSV missing columns: {missing}")

    base_output = args.output or args.csv.with_name(args.csv.stem + "_depth_perf.png")
    stats = compute_bin_stats(df)
    if stats is None:
        n_touch = int((df["label"] == 1).sum())
        n_with_depth = int(df["depth_touch"].notna().sum())
        print(
            f"[SKIP] depth eval — no touch-positive rows have depth_touch populated.\n"
            f"       touch-positive rows: {n_touch}  |  rows with depth_touch: {n_with_depth}"
        )
        return

    metrics = _METRICS if args.all_metrics else [args.metric]
    for metric in metrics:
        metric_label = "Recall (hit rate)" if metric == "recall" else "Accuracy"
        out = plot_style.with_metric_suffix(base_output, metric) if args.all_metrics else base_output
        plot_depth_bars(stats, metric_label, out)
        print(f"Saved → {out}")

    print(f"\nPer-bin stats:\n{stats[['bin_label', 'n', 'recall']].to_string(index=False)}")

    print(f"\nGlobal stats (all samples, n={len(df)}):")
    touch = df.dropna(subset=["depth_touch"])
    print(f"\nGlobal stats (depth-annotated samples, n={len(touch)}):")
    if len(touch) > 0:
        gs = global_stats(touch)
        print(f"\nGlobal (depth-annotated samples, n={gs['total']}): "
              f"accuracy={gs['accuracy']:.3f}  F1={gs['f1']:.3f}\n"
            f"  Precision: {gs['tp'] / (gs['tp'] + gs['fp']):.3f}\n"
            f"  Recall   : {gs['tp'] / (gs['tp'] + gs['fn']):.3f}\n"
        )


def global_stats(df: pd.DataFrame) -> dict:
    y_true, y_pred = df["label"].astype(int), df["prediction"].astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {"accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "total": len(df)}


if __name__ == "__main__":
    main()
