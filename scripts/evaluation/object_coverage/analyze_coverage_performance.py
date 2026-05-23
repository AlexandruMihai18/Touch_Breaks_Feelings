"""Analyse model touch-detection performance across object-coverage bins.

Unlike the depth and zone analyses (which only have touch-positive samples per
bin), object_coverage is available for ALL samples — so this script computes
full F1 and accuracy per bin, not just recall.

Reads a predictions CSV with a `object_coverage` column and bins samples by
quantile, then renders a grouped bar chart of F1 and accuracy per bin.

Usage
-----
    python scripts/evaluation/object_coverage/analyze_coverage_performance.py \\
        results/predictions.csv \\
        [--n-bins 5] [--metric f1|accuracy] [--output coverage_performance.png]
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
    valid = df.dropna(subset=["object_coverage"]).copy()
    if len(valid) == 0:
        raise ValueError("No rows with object_coverage found in CSV.")

    valid["bin"], bin_edges = pd.qcut(
        valid["object_coverage"], q=n_bins, retbins=True, duplicates="drop"
    )

    rows = []
    for interval, group in valid.groupby("bin", observed=True):
        y_true = group["label"].astype(int)
        y_pred = group["prediction"].astype(int)
        n      = len(group)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        rows.append({
            "bin_label":  f"{interval.left:.3f}–{interval.right:.3f}",
            "cov_mid":    (interval.left + interval.right) / 2,
            "n":          n,
            "n_touch":    int((y_true == 1).sum()),
            "accuracy":   accuracy_score(y_true, y_pred),
            "f1":         f1_score(y_true, y_pred, zero_division=0),
            "recall":     tp / (tp + fn) if (tp + fn) > 0 else float("nan"),
            "precision":  tp / (tp + fp) if (tp + fp) > 0 else float("nan"),
        })

    return pd.DataFrame(rows).sort_values("cov_mid")


def plot_bars(stats: pd.DataFrame, metric: str, output: Path) -> None:
    plot_style.apply()

    n = len(stats)
    fig, ax = plt.subplots(figsize=(max(5.5, n * 1.1), 4.0))

    cmap = plt.cm.RdYlGn
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    vals = stats[metric].values
    colors = [cmap(norm(v)) for v in vals]

    ax.bar(
        range(n), vals,
        color=colors,
        edgecolor="white",
        linewidth=0.6,
        width=0.58,
        zorder=3,
    )

    for i, (_, row) in enumerate(stats.iterrows()):
        ax.text(
            i, row[metric] + 0.025,
            f"{row[metric]:.2f}",
            ha="center", va="bottom", fontsize=9, fontweight="semibold",
            color=plot_style.DARK,
        )
        ax.text(
            i, -0.055,
            f"n={row['n']}  ({row['n_touch']} touch)",
            ha="center", va="top", fontsize=7.0, color=plot_style.GRAY,
            transform=ax.get_xaxis_transform(),
        )

    mean_val = stats[metric].mean()
    ax.axhline(
        mean_val, color=plot_style.DARK, linestyle="--", linewidth=1.0,
        label=f"Mean {metric} = {mean_val:.2f}", zorder=4,
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.03, aspect=25)
    cbar.set_label(metric.capitalize(), fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_linewidth(0.5)

    ax.set_xticks(range(n))
    ax.set_xticklabels(stats["bin_label"], rotation=30, ha="right", fontsize=9)
    ax.set_xlabel("Object coverage bin  (foreground pixels / total pixels)", labelpad=8)
    ax.set_ylabel(metric.capitalize())
    ax.set_ylim(0, 1.18)
    ax.set_xlim(-0.55, n - 0.45)
    ax.set_title(f"Touch-detection {metric.upper()} by object coverage bin\n(all samples — touch and no-touch)")
    ax.legend(loc="upper left", fontsize=9)

    plt.savefig(output)
    plt.close(fig)


def global_stats(df: pd.DataFrame) -> dict:
    y_true = df["label"].astype(int)
    y_pred = df["prediction"].astype(int)
    cm     = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1":       f1_score(y_true, y_pred, zero_division=0),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "total": len(df),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--n-bins", type=int,    default=5,        help="Number of quantile coverage bins (default: 5)")
    parser.add_argument("--metric", choices=["recall", "precision", "f1", "accuracy"],
                        default="f1", help="Metric to display per bin (default: f1)")
    parser.add_argument("--output", type=Path,   default=None)
    parser.add_argument("--annotations", nargs="+", type=Path, required=True,
                        help="Annotation JSON file(s) providing object_coverage (required; not in prediction CSV)")
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(args.csv)

    df = pd.read_csv(args.csv)
    df = enrich_df(df, args.annotations)

    if missing := {"label", "prediction", "object_coverage"} - set(df.columns):
        raise ValueError(f"CSV missing columns: {missing}")

    output = args.output or args.csv.with_name(args.csv.stem + "_coverage_perf.png")
    stats  = compute_bin_stats(df, args.n_bins)
    plot_bars(stats, args.metric, output)
    print(f"Saved → {output}")
    all_metrics = ["recall", "precision", "f1", "accuracy"]
    print(f"\nPer-bin {args.metric}:\n{stats[['bin_label', 'n', 'n_touch'] + all_metrics].to_string(index=False, float_format=lambda v: f'{v:.3f}')}")

    gs = global_stats(df.dropna(subset=["object_coverage"]))
    print(f"\nGlobal (coverage-annotated samples, n={gs['total']}): "
          f"accuracy={gs['accuracy']:.3f}  F1={gs['f1']:.3f}  "
          f"TP={gs['tp']}  FP={gs['fp']}  FN={gs['fn']}  TN={gs['tn']}")


if __name__ == "__main__":
    main()
