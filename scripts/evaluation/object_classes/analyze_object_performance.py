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
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import enrich_df


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

def plot_cluster_bars(stats: pd.DataFrame, output: Path) -> None:
    clusters = stats["cluster"].tolist()
    x = np.arange(len(clusters))
    width = 0.2

    fig, (ax_metrics, ax_counts) = plt.subplots(
        2, 1, figsize=(max(8, len(clusters) * 1.4), 10),
        gridspec_kw={"height_ratios": [1.2, 1]},
    )

    # — top: recall / precision / F1 per cluster —
    ax_metrics.bar(x - width, stats["recall"],    width, label="Recall",    color="#4e9af1")
    ax_metrics.bar(x,          stats["precision"], width, label="Precision", color="#f18f4e")
    ax_metrics.bar(x + width,  stats["f1"],        width, label="F1",       color="#6abf69")

    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels(clusters, rotation=30, ha="right", fontsize=9)
    ax_metrics.set_ylim(0, 1.2)
    ax_metrics.set_ylabel("Score", fontsize=10)
    ax_metrics.set_title("Per-cluster metrics", fontsize=12, fontweight="bold")
    ax_metrics.legend(fontsize=9)
    ax_metrics.spines[["top", "right"]].set_visible(False)

    for i, (_, row) in enumerate(stats.iterrows()):
        ax_metrics.text(i - width, row["recall"]    + 0.02, f"{row['recall']:.2f}",    ha="center", va="bottom", fontsize=7)
        ax_metrics.text(i,          row["precision"] + 0.02, f"{row['precision']:.2f}", ha="center", va="bottom", fontsize=7)
        ax_metrics.text(i + width,  row["f1"]        + 0.02, f"{row['f1']:.2f}",        ha="center", va="bottom", fontsize=7)

    # — bottom: TP / TN / FP / FN stacked counts —
    ax_counts.bar(x, stats["tp"], width * 2.5, label="TP", color="#6abf69")
    ax_counts.bar(x, stats["tn"], width * 2.5, label="TN", color="#4e9af1",  bottom=stats["tp"])
    ax_counts.bar(x, stats["fp"], width * 2.5, label="FP", color="#f4a043",  bottom=stats["tp"] + stats["tn"])
    ax_counts.bar(x, stats["fn"], width * 2.5, label="FN", color="#e05a5a",  bottom=stats["tp"] + stats["tn"] + stats["fp"])

    ax_counts.set_xticks(x)
    ax_counts.set_xticklabels(clusters, rotation=30, ha="right", fontsize=9)
    ax_counts.set_ylabel("Sample count", fontsize=10)
    ax_counts.set_title("Confusion counts per cluster", fontsize=12, fontweight="bold")
    ax_counts.legend(fontsize=9)
    ax_counts.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
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
    plot_cluster_bars(stats, output)
    print(f"Saved → {output}")

    cols = ["cluster", "n", "tp", "tn", "fp", "fn", "recall", "precision", "f1", "accuracy"]
    print(f"\nPer-cluster results (n={len(df)}):\n")
    print(stats[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # Global
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
