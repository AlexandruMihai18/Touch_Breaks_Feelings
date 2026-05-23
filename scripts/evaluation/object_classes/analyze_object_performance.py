"""Analyze model touch-detection performance per object cluster.

Reads a predictions CSV, joins object_name from annotation JSONs, maps each
object to a semantic cluster via a pre-built cluster JSON, then computes
TP / TN / FP / FN (and derived metrics) per cluster.

Usage
-----
    python scripts/evaluation/object_classes/analyze_object_performance.py \\
        results/predictions.csv \\
        --clusters  data/epic_kitchen/object_clusters_7.json \\
        --annotations /scratch-shared/dotero/epic_kitchen/annotations/val.json \\
        [--output results/eval_object_perf.png]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import enrich_df
import plot_style


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def compute_cluster_stats(df: pd.DataFrame, cluster_map: dict[str, str]) -> pd.DataFrame:
    valid = df.dropna(subset=["object_name"]).copy()
    if len(valid) == 0:
        return None

    valid["cluster"] = valid["object_name"].map(cluster_map).fillna("Unknown")

    rows = []
    for cluster, group in valid.groupby("cluster", sort=False):
        y_true = group["label"].astype(int)
        y_pred = group["prediction"].astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        n = len(group)
        rows.append({
            "cluster":   cluster,
            "n":         n,
            "tp":        int(tp),
            "tn":        int(tn),
            "fp":        int(fp),
            "fn":        int(fn),
            "recall":    tp / (tp + fn) if (tp + fn) > 0 else float("nan"),
            "precision": tp / (tp + fp) if (tp + fp) > 0 else float("nan"),
            "f1":        f1_score(y_true, y_pred, zero_division=0),
            "accuracy":  accuracy_score(y_true, y_pred),
        })

    return pd.DataFrame(rows).sort_values("n", ascending=False)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_cluster_bars(stats: pd.DataFrame, metric: str, output: Path) -> None:
    plot_style.apply()

    clusters = stats["cluster"].tolist()
    x = np.arange(len(clusters))

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        figsize=(max(7.5, len(clusters) * 1.3), 8.5),
        gridspec_kw={"height_ratios": [1.1, 1.0]},
    )

    # ── Top: selected metric, one bar per cluster, RdYlGn coloured ───────────
    cmap = plt.cm.RdYlGn
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    vals = stats[metric].fillna(0).values
    colors = [cmap(norm(v)) for v in vals]

    ax_top.bar(x, vals, width=0.58, color=colors, edgecolor="white", linewidth=0.6, zorder=3)

    for i, (_, row) in enumerate(stats.iterrows()):
        v = row[metric]
        if not np.isnan(v):
            ax_top.text(i, v + 0.025, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8, fontweight="semibold",
                        color=plot_style.DARK)
        ax_top.text(i, -0.055, f"n={row['n']}", ha="center", va="top",
                    fontsize=7.5, color=plot_style.GRAY,
                    transform=ax_top.get_xaxis_transform())

    mean_val = np.nanmean(vals)
    ax_top.axhline(mean_val, color=plot_style.DARK, linestyle="--", linewidth=1.0,
                   label=f"Mean {metric} = {mean_val:.2f}", zorder=4)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_top, pad=0.01, fraction=0.025, aspect=25)
    cbar.set_label(metric.capitalize(), fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_linewidth(0.5)

    ax_top.set_xticks(x)
    ax_top.set_xticklabels(clusters, rotation=30, ha="right", fontsize=9)
    ax_top.set_ylim(0, 1.18)
    ax_top.set_ylabel(metric.capitalize())
    ax_top.set_title(f"Touch-detection {metric} by object cluster")
    ax_top.legend(loc="upper left", fontsize=9)
    ax_top.set_xlim(-0.55, len(clusters) - 0.45)

    # ── Bottom: TP / TN / FP / FN proportional stacked bars ──────────────────
    totals = stats[["tp", "tn", "fp", "fn"]].sum(axis=1).values
    tp_p = stats["tp"].values / totals
    tn_p = stats["tn"].values / totals
    fp_p = stats["fp"].values / totals
    fn_p = stats["fn"].values / totals

    bar_w = 0.55
    ax_bot.bar(x, tp_p,                         width=bar_w, label="TP", color=plot_style.C_TP, zorder=3)
    ax_bot.bar(x, tn_p, bottom=tp_p,            width=bar_w, label="TN", color=plot_style.C_TN, zorder=3)
    ax_bot.bar(x, fp_p, bottom=tp_p + tn_p,     width=bar_w, label="FP", color=plot_style.C_FP, zorder=3)
    ax_bot.bar(x, fn_p, bottom=tp_p + tn_p + fp_p, width=bar_w, label="FN", color=plot_style.C_FN, zorder=3)

    for i, (_, row) in enumerate(stats.iterrows()):
        ax_bot.text(i, -0.06, f"n={row['n']}", ha="center", va="top",
                    fontsize=7.5, color=plot_style.GRAY,
                    transform=ax_bot.get_xaxis_transform())

    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(clusters, rotation=30, ha="right", fontsize=9)
    ax_bot.set_ylim(0, 1.18)
    ax_bot.set_ylabel("Proportion of samples")
    ax_bot.set_title("Confusion breakdown per cluster  (proportional)")
    ax_bot.legend(loc="upper right", ncol=4)
    ax_bot.set_xlim(-0.6, len(clusters) - 0.4)
    ax_bot.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    plt.savefig(output)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv",        type=Path, help="Predictions CSV")
    parser.add_argument("--clusters", type=Path, required=True,
                        help="Cluster mapping JSON produced by cluster_object_classes.py")
    parser.add_argument("--annotations", nargs="+", type=Path, default=None,
                        help="Annotation JSON file(s) to join object_name into the CSV")
    parser.add_argument("--metric", choices=["recall", "precision", "f1", "accuracy"],
                        default="f1", help="Metric shown in the bar chart (default: f1)")
    parser.add_argument("--output",   type=Path, default=None,
                        help="Output PNG path (default: <csv_stem>_object_perf.png)")
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(args.csv)
    if not args.clusters.exists():
        raise FileNotFoundError(args.clusters)

    df = pd.read_csv(args.csv)
    if args.annotations:
        df = enrich_df(df, args.annotations)

    if "object_name" not in df.columns or df["object_name"].isna().all():
        print("[SKIP] object eval — object_name not populated. Run enrich_df with annotation JSONs.")
        return

    with open(args.clusters) as f:
        cluster_map: dict[str, str] = json.load(f)

    stats = compute_cluster_stats(df, cluster_map)
    if stats is None or len(stats) == 0:
        print("[SKIP] object eval — no rows with object_name after enrichment.")
        return

    output = args.output or args.csv.with_name(args.csv.stem + "_object_perf.png")
    plot_cluster_bars(stats, args.metric, output)
    print(f"Saved → {output}")

    cols = ["cluster", "n", "tp", "tn", "fp", "fn", "recall", "precision", "f1", "accuracy"]
    print(f"\nPer-cluster results (n={len(df)}):\n")
    print(stats[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    y_true = df["label"].astype(int)
    y_pred = df["prediction"].astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print(
        f"\nGlobal (all {len(df)} rows): "
        f"TP={tp}  TN={tn}  FP={fp}  FN={fn}  "
        f"F1={f1_score(y_true, y_pred, zero_division=0):.3f}  "
        f"acc={accuracy_score(y_true, y_pred):.3f}"
    )


if __name__ == "__main__":
    main()
