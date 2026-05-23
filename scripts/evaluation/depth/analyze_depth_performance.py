"""Analyze model touch-detection performance across depth bins.

Reads a prediction CSV that must include a `depth_touch` column (median grayscale
depth under the touch mask, populated by annotate_touch_depth.py).  Bins the depth
values and computes per-bin recall (hit rate on touch-positive samples).

Usage
-----
    python scripts/analyze_depth_performance.py results/predictions.csv \\
        [--n-bins 5] [--output depth_performance.png]
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


def compute_bin_stats(df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    """Bin depth values by quantile and compute per-bin recall on touch samples."""
    touch = df.dropna(subset=["depth_touch"]).copy()
    touch = touch[touch["label"] == 1]

    if len(touch) == 0:
        return None

    touch["bin"], bin_edges = pd.qcut(
        touch["depth_touch"], q=n_bins, retbins=True, duplicates="drop"
    )

    rows = []
    for interval, group in touch.groupby("bin", observed=True):
        n    = len(group)
        hits = (group["prediction"] == 1).sum()
        rows.append({
            "bin_label": f"{interval.left:.0f}–{interval.right:.0f}",
            "depth_mid": (interval.left + interval.right) / 2,
            "n":         n,
            "recall":    hits / n,
        })

    if not rows:
        return None

    return pd.DataFrame(rows).sort_values("depth_mid")


def plot_depth_bars(stats: pd.DataFrame, metric_label: str, output_path: Path) -> None:
    plot_style.apply()

    n = len(stats)
    fig, ax = plt.subplots(figsize=(max(5.5, n * 1.1), 4.0))

    cmap = plt.cm.RdYlGn
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    colors = [cmap(norm(v)) for v in stats["recall"]]

    ax.bar(
        range(n),
        stats["recall"],
        color=colors,
        edgecolor="white",
        linewidth=0.6,
        width=0.58,
        zorder=3,
    )

    # Value + count labels above each bar
    for i, (_, row) in enumerate(stats.iterrows()):
        ax.text(
            i, row["recall"] + 0.025,
            f"{row['recall']:.2f}",
            ha="center", va="bottom", fontsize=9, fontweight="semibold",
            color=plot_style.DARK,
        )
        ax.text(
            i, -0.055,
            f"n={row['n']}",
            ha="center", va="top", fontsize=7.5, color=plot_style.GRAY,
            transform=ax.get_xaxis_transform(),
        )

    mean_val = stats["recall"].mean()
    ax.axhline(
        mean_val, color=plot_style.DARK, linestyle="--", linewidth=1.0,
        label=f"Mean {metric_label.lower()} = {mean_val:.2f}", zorder=4,
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.03, aspect=25)
    cbar.set_label(metric_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_linewidth(0.5)

    ax.set_xticks(range(n))
    ax.set_xticklabels(stats["bin_label"], rotation=30, ha="right", fontsize=9)
    ax.set_xlabel(
        "Depth bin  (disparity value — low = far,  high = near)",
        labelpad=8,
    )
    ax.set_ylabel(metric_label)
    ax.set_ylim(0, 1.18)
    ax.set_xlim(-0.55, n - 0.45)
    ax.set_title(f"Touch-detection {metric_label.lower()} by depth bin\n(touch-positive samples only)")
    ax.legend(loc="upper left", fontsize=9)

    plt.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--n-bins", type=int, default=5,
                        help="Number of quantile depth bins (default: 5)")
    parser.add_argument("--metric", choices=["recall", "accuracy"], default="recall",
                        help="Metric label shown on the figure (default: recall). "
                             "Both are equivalent on touch-only samples; choose the label you prefer.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--annotations", nargs="+", type=Path, required=True,
                        help="Annotation JSON file(s) providing depth_touch (required; not in prediction CSV)")
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(args.csv)

    df = pd.read_csv(args.csv)
    df = enrich_df(df, args.annotations)

    required = {"label", "prediction", "depth_touch"}
    if missing := required - set(df.columns):
        raise ValueError(f"CSV missing columns: {missing}")

    metric_label = "Recall (hit rate)" if args.metric == "recall" else "Accuracy"
    output = args.output or args.csv.with_name(args.csv.stem + "_depth_perf.png")
    stats  = compute_bin_stats(df, args.n_bins)
    if stats is None:
        n_touch = int((df["label"] == 1).sum())
        n_with_depth = int(df["depth_touch"].notna().sum())
        print(
            f"[SKIP] depth eval — no touch-positive rows have depth_touch populated.\n"
            f"       touch-positive rows: {n_touch}  |  rows with depth_touch: {n_with_depth}\n"
            f"       annotate_touch_depth.py may not support this dataset's depth format."
        )
        return
    plot_depth_bars(stats, metric_label, output)
    print(f"Saved → {output}")
    print(f"\nPer-bin {args.metric}:\n{stats[['bin_label', 'n', 'recall']].to_string(index=False)}")

    touch = df.dropna(subset=["depth_touch"])
    if len(touch) > 0:
        gs = global_stats(touch)
        print(f"\nGlobal (depth-annotated samples only, n={gs['total']}): "
              f"accuracy={gs['accuracy']:.3f}  F1={gs['f1']:.3f}")


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
