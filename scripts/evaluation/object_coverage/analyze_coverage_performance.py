"""Analyse model touch-detection performance across object-coverage bins.

Unlike the depth and zone analyses (which only have touch-positive samples per
bin), object_coverage is available for ALL samples — so this script computes
full metrics per bin.

Coverage thresholds (nonzero pixels / total pixels):
    < 0.02           → Small        (utensils, small produce: knife, spoon, carrot, egg)
    0.02 – 0.07      → Medium-small (cookware, containers: bowl, pan, kettle, plate)
    0.07 – 0.15      → Medium-large (boards, fixed appliances: chopping board, dishwasher)
    ≥ 0.15           → Large        (surfaces: oven, hob, sink, fridge, cupboard)

Usage
-----
    python scripts/evaluation/object_coverage/analyze_coverage_performance.py \\
        results/predictions.csv \\
        --annotations /path/to/annotations.json \\
        [--metric f1] [--output coverage_performance.png]
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


# Hard coverage thresholds derived from EPIC-Kitchens object_labels_coverage.txt
_BINS = [
    ("Small",         0.00,  0.02),
    ("Medium-small",  0.02,  0.07),
    ("Medium-large",  0.07,  0.15),
    ("Large",         0.15,  1.01),
]

# Viridis sampled at 4 evenly spaced points — colorblind-safe, print-friendly
_BIN_COLORS = [plt.cm.viridis(v) for v in (0.10, 0.38, 0.65, 0.90)]


def compute_bin_stats(df: pd.DataFrame) -> pd.DataFrame:
    valid = df.dropna(subset=["object_coverage"]).copy()
    if len(valid) == 0:
        raise ValueError("No rows with object_coverage found in CSV.")

    rows = []
    for label, lo, hi in _BINS:
        group = valid[(valid["object_coverage"] >= lo) & (valid["object_coverage"] < hi)]
        if len(group) == 0:
            continue
        y_true = group["label"].astype(int)
        y_pred = group["prediction"].astype(int)
        n = len(group)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        rows.append({
            "bin_label":  label,
            "n":          n,
            "n_touch":    int((y_true == 1).sum()),
            "accuracy":   accuracy_score(y_true, y_pred),
            "f1":         f1_score(y_true, y_pred, zero_division=0),
            "recall":     tp / (tp + fn) if (tp + fn) > 0 else float("nan"),
            "precision":  tp / (tp + fp) if (tp + fp) > 0 else float("nan"),
        })

    return pd.DataFrame(rows)


def plot_bars(stats: pd.DataFrame, metric: str, output: Path) -> None:
    plot_style.apply()

    n = len(stats)
    colors = _BIN_COLORS[:n]

    fig, ax = plt.subplots(figsize=(3.5 + n * 0.9, 4.0))

    vals = stats[metric].values
    ax.bar(
        range(n), vals,
        color=colors,
        edgecolor="white",
        linewidth=0.6,
        width=0.52,
        zorder=3,
    )

    for i, (_, row) in enumerate(stats.iterrows()):
        v = row[metric]
        if not np.isnan(v):
            ax.text(
                i, v + 0.025,
                f"{v:.2f}",
                ha="center", va="bottom", fontsize=11, fontweight="semibold",
                color=plot_style.DARK,
            )

    mean_val = np.nanmean(vals)
    ax.axhline(
        mean_val, color=plot_style.GRAY, linestyle="--", linewidth=1.0,
        label=f"Mean = {mean_val:.2f}", zorder=4,
    )

    ax.set_xticks(range(n))
    ax.set_xticklabels(stats["bin_label"], fontsize=11)
    ax.set_ylabel(metric.capitalize())
    ax.set_ylim(0, 1.15)
    ax.set_xlim(-0.55, n - 0.45)
    ax.set_title(f"{metric.upper()} by object size")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlabel("")

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
    _METRICS = ["recall", "precision", "f1", "accuracy"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--metric", choices=_METRICS, default="f1",
                        help="Metric to display per bin (default: f1)")
    parser.add_argument("--all-metrics", action="store_true",
                        help=f"Generate one figure per metric {_METRICS}, each suffixed with the metric name. "
                             "Overrides --metric.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--annotations", nargs="+", type=Path, required=True,
                        help="Annotation JSON file(s) providing object_coverage")
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(args.csv)

    df = pd.read_csv(args.csv)
    df = enrich_df(df, args.annotations)

    if missing := {"label", "prediction", "object_coverage"} - set(df.columns):
        raise ValueError(f"CSV missing columns: {missing}")

    base_output = args.output or args.csv.with_name(args.csv.stem + "_coverage_perf.png")
    stats = compute_bin_stats(df)

    metrics = _METRICS if args.all_metrics else [args.metric]
    for metric in metrics:
        out = plot_style.with_metric_suffix(base_output, metric) if args.all_metrics else base_output
        plot_bars(stats, metric, out)
        print(f"Saved → {out}")

    print(f"\nPer-bin stats:\n{stats[['bin_label', 'n', 'n_touch'] + _METRICS].to_string(index=False, float_format=lambda v: f'{v:.3f}')}")

    gs = global_stats(df.dropna(subset=["object_coverage"]))
    print(f"\nGlobal (coverage-annotated samples, n={gs['total']}): "
          f"accuracy={gs['accuracy']:.3f}  F1={gs['f1']:.3f}  "
          f"TP={gs['tp']}  FP={gs['fp']}  FN={gs['fn']}  TN={gs['tn']}")


if __name__ == "__main__":
    main()
