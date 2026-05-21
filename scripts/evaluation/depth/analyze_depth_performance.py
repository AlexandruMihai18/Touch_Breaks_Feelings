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
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix


def compute_bin_stats(df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    """Bin depth values by quantile and compute per-bin recall on touch samples."""
    touch = df.dropna(subset=["depth_touch"]).copy()
    touch = touch[touch["label"] == 1]

    if len(touch) == 0:
        raise ValueError("No touch-positive rows with depth_touch found in CSV.")

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

    return pd.DataFrame(rows).sort_values("depth_mid")


def plot_depth_bars(stats: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(stats) * 1.2), 5))

    cmap  = plt.cm.RdYlGn
    norm  = plt.Normalize(0, 1)
    bars  = ax.bar(
        range(len(stats)),
        stats["recall"],
        color=[cmap(norm(v)) for v in stats["recall"]],
        edgecolor="white",
        linewidth=0.8,
        width=0.6,
    )

    for i, (_, row) in enumerate(stats.iterrows()):
        ax.text(i, row["recall"] + 0.02, f"{row['recall']:.2f}\nn={row['n']}",
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(stats)))
    ax.set_xticklabels(stats["bin_label"], rotation=30, ha="right", fontsize=9)
    ax.set_xlabel(
        "Depth bin  (inferno colormap of Depth-Anything-V2 disparity — low = dark/purple = far,  high = bright/yellow = near)",
        fontsize=9,
    )
    ax.set_ylabel("Recall (hit rate)", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_title("Per-depth-bin recall\n(touch-positive samples only)", fontsize=12, fontweight="bold")
    ax.axhline(stats["recall"].mean(), color="black", linestyle="--", linewidth=1,
               label=f"mean recall = {stats['recall'].mean():.2f}")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--n-bins", type=int, default=5,
                        help="Number of quantile depth bins (default: 5)")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(args.csv)

    df = pd.read_csv(args.csv)
    required = {"label", "prediction", "depth_touch"}
    if missing := required - set(df.columns):
        raise ValueError(f"CSV missing columns: {missing}")

    output = args.output or args.csv.with_name(args.csv.stem + "_depth_perf.png")
    stats  = compute_bin_stats(df, args.n_bins)
    plot_depth_bars(stats, output)
    print(f"Saved → {output}")
    print(f"\nPer-bin recall:\n{stats[['bin_label', 'n', 'recall']].to_string(index=False)}")

    # Global stats over depth-annotated touch samples
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
