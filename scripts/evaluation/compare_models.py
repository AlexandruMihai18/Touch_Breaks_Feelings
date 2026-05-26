"""Compare dino_mlp3, seggpt, and qwen_vision across datasets.

Generates three comparison figures:
  depth_comparison.png   — recall by depth bin, per dataset, all 3 models
  class_comparison.png   — recall by object cluster (Epic Kitchen only), all 3 models
  zone_comparison.png    — touch zone heatmaps (3 models × 3 datasets), shared colorbar

Usage
-----
    # Default cluster paths
    python scripts/evaluation/compare_models.py --output-dir results/comparison

    # Override annotation paths (e.g. on local machine)
    python scripts/evaluation/compare_models.py \\
        --ek-annotations   data/epic_kitchen/annotations/val.json \\
        --gh-annotations   data/greatest_hits/annotations/val.json \\
        --output-dir       results/comparison
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from utils import enrich_df
import plot_style

# ── Model registry ────────────────────────────────────────────────────────────

_MODELS = [
    {
        "name":  "dino_mlp3",
        "label": "DINOv2-MLP3",
        "color": plot_style.C_RECALL,   # blue
        "csvs": {
            "epic_kitchen": "/home/dotero/Touch_Breaks_Feelings/results/ek_dino_mlp3_prediction.csv",
            "greatest_hits": "/home/dotero/Touch_Breaks_Feelings/results/gh_dino_mlp3_prediction.csv",
            "kubric":       "/home/dotero/Touch_Breaks_Feelings/results/kubric_dino_mlp3_prediction.csv",
        },
    },
    {
        "name":  "seggpt",
        "label": "SegGPT",
        "color": plot_style.C_PREC,     # orange
        "csvs": {
            "epic_kitchen": "/home/dotero/Touch_Breaks_Feelings/results/epic_kitchen_val_seggpt_touch_results.csv",
            "greatest_hits": "/home/dotero/Touch_Breaks_Feelings/results/greatest_hits_val_seggpt_touch_results.csv",
            "kubric":       "/home/dotero/Touch_Breaks_Feelings/results/kubric_val_seggpt_touch_results.csv",
        },
    },
    {
        "name":  "qwen_vision",
        "label": "Qwen3-VL",
        "color": plot_style.C_F1,       # green
        "csvs": {
            "epic_kitchen": "/home/dotero/Touch_Breaks_Feelings/results/EpicKitchen_qwen3_visual_baseline_predictions.csv",
            "greatest_hits": "/home/dotero/Touch_Breaks_Feelings/results/GreatestHits_qwen3_visual_baseline_predictions.csv",
            "kubric":       "/home/dotero/Touch_Breaks_Feelings/results/Kubric_qwen3_visual_baseline_predictions.csv",
        },
    },
]

# ── Dataset registry ──────────────────────────────────────────────────────────

_DATASETS = [
    {
        "name":  "epic_kitchen",
        "label": "Epic Kitchen",
        "annotations_cluster": "/scratch-shared/dotero/epic_kitchen/annotations/val.json",
        "annotations_local":   str(_REPO / "data/epic_kitchen/annotations/val.json"),
        "clusters_cluster":    "/home/dotero/Touch_Breaks_Feelings/data/epic_kitchen/object_clusters_7.json",
        "clusters_local":      str(_REPO / "data/epic_kitchen/object_clusters_7.json"),
    },
    {
        "name":  "greatest_hits",
        "label": "Greatest Hits",
        "annotations_cluster": "/scratch-shared/dotero/greatest_hits/annotations/val.json",
        "annotations_local":   str(_REPO / "data/greatest_hits/annotations/val.json"),
        "clusters_cluster":    None,
        "clusters_local":      None,
    },
    {
        "name":  "kubric",
        "label": "Kubric",
        "annotations_cluster": "/scratch-shared/dotero/kubric/annotations/val.json",
        "annotations_local":   None,
        "clusters_cluster":    None,
        "clusters_local":      None,
    },
]

# Depth bins: (label, lo_inclusive, hi_exclusive)
_DEPTH_BINS = [
    ("Far",    0,   85),
    ("Medium", 85,  170),
    ("Close",  170, 256),
]


# ── Data loading ──────────────────────────────────────────────────────────────

def _resolve_annotations(ds: dict, overrides: dict[str, str]) -> list[Path]:
    name = ds["name"]
    if name in overrides:
        p = Path(overrides[name])
        return [p] if p.exists() else []
    for key in ("annotations_cluster", "annotations_local"):
        v = ds.get(key)
        if v and Path(v).exists():
            return [Path(v)]
    return []


def _resolve_clusters(ds: dict) -> Path | None:
    for key in ("clusters_cluster", "clusters_local"):
        v = ds.get(key)
        if v and Path(v).exists():
            return Path(v)
    return None


def load_df(csv_path: str, annotation_paths: list[Path]) -> pd.DataFrame | None:
    p = Path(csv_path)
    if not p.exists():
        print(f"  [SKIP] CSV not found: {p}")
        return None
    df = pd.read_csv(p)
    if annotation_paths:
        df = enrich_df(df, annotation_paths)
    else:
        print(f"  [WARN] no annotations found for {p.name}, CSV not enriched")
    return df


# ── Metric computation ────────────────────────────────────────────────────────

def depth_recall(df: pd.DataFrame) -> dict[str, float]:
    """Recall per depth bin over touch-positive samples with depth_touch set."""
    touch = df.dropna(subset=["depth_touch"])
    touch = touch[touch["label"] == 1]
    result: dict[str, float] = {}
    for label, lo, hi in _DEPTH_BINS:
        grp = touch[(touch["depth_touch"] >= lo) & (touch["depth_touch"] < hi)]
        result[label] = float((grp["prediction"] == 1).sum() / len(grp)) if len(grp) > 0 else float("nan")
    return result


def cluster_recall(df: pd.DataFrame, cluster_map: dict[str, str]) -> dict[str, dict]:
    """Recall + n per semantic cluster."""
    valid = df.dropna(subset=["object_name"]).copy()
    if len(valid) == 0:
        return {}
    valid["cluster"] = valid["object_name"].map(cluster_map).fillna("Unknown")
    result: dict[str, dict] = {}
    for cluster, grp in valid.groupby("cluster"):
        tp = int(((grp["label"] == 1) & (grp["prediction"] == 1)).sum())
        fn = int(((grp["label"] == 1) & (grp["prediction"] == 0)).sum())
        result[cluster] = {
            "n":      len(grp),
            "recall": tp / (tp + fn) if (tp + fn) > 0 else float("nan"),
        }
    return result


def zone_recall(df: pd.DataFrame, grid: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Per-zone recall and count grids, shaped (grid, grid)."""
    recall = np.full((grid, grid), np.nan)
    count  = np.zeros((grid, grid), dtype=int)

    touch = df.dropna(subset=["x_touch", "y_touch"]).copy()
    touch["x_touch"] = touch["x_touch"].astype(int)
    touch["y_touch"] = touch["y_touch"].astype(int)
    touch = touch[touch["x_touch"].between(0, grid - 1) & touch["y_touch"].between(0, grid - 1)]

    for (y, x), grp in touch.groupby(["y_touch", "x_touch"]):
        count[y, x] = len(grp)
        recall[y, x] = (grp["prediction"] == grp["label"]).sum() / len(grp)

    return recall, count


# ── Plot 1: Depth comparison (one figure per dataset) ─────────────────────────

def plot_depth_dataset(
    model_bins: dict[str, dict[str, float]],
    dataset_label: str,
    output: Path,
) -> None:
    """
    model_bins: {model_name → {bin_label → recall}}
    Skips the figure entirely if no model has any non-NaN depth value.
    """
    has_data = any(
        not np.isnan(v)
        for bins in model_bins.values()
        for v in bins.values()
    )
    if not has_data:
        print(f"[SKIP] depth — no depth_touch data for {dataset_label}")
        return

    plot_style.apply()
    bin_labels = [b[0] for b in _DEPTH_BINS]
    n_models   = len(_MODELS)
    bar_w      = 0.22
    offsets    = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * bar_w
    x          = np.arange(len(bin_labels))

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    for model, offset in zip(_MODELS, offsets):
        bins = model_bins.get(model["name"], {})
        ys = [bins.get(b, float("nan")) for b in bin_labels]
        ys_plot = [0.0 if np.isnan(v) else v for v in ys]

        ax.bar(
            x + offset, ys_plot,
            width=bar_w * 0.92,
            color=model["color"],
            label=model["label"],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        for j, v in enumerate(ys):
            if not np.isnan(v):
                ax.text(
                    x[j] + offset, v + 0.018,
                    f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7,
                    color=plot_style.DARK,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, fontsize=10)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Recall")
    ax.set_title(f"Touch Recall by Depth Distance\n{dataset_label}",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(-0.7, len(bin_labels) - 0.3)
    ax.legend(loc="upper left", fontsize=8)

    plt.savefig(output)
    plt.close(fig)
    print(f"Saved → {output}")


# ── Plot 2: Object class comparison (Epic Kitchen) ────────────────────────────

def plot_class_comparison(
    class_data: dict[str, dict[str, dict]],
    output: Path,
) -> None:
    """
    class_data: {model_name → {cluster → {"n": int, "recall": float}}}
    """
    # Aggregate cluster sizes across models to determine order
    totals: dict[str, int] = {}
    for model_dict in class_data.values():
        for cluster, stats in model_dict.items():
            totals[cluster] = totals.get(cluster, 0) + stats["n"]

    if not totals:
        print("[SKIP] class comparison — no cluster data for Epic Kitchen")
        return

    clusters = sorted(totals, key=lambda c: totals[c], reverse=True)
    clusters = [c for c in clusters if c != "Unknown"]

    plot_style.apply()
    n_c      = len(clusters)
    n_models = len(_MODELS)
    bar_w    = 0.22
    offsets  = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * bar_w
    x        = np.arange(n_c)

    fig, ax = plt.subplots(figsize=(max(9.0, n_c * 1.5), 4.5))

    for i, (model, offset) in enumerate(zip(_MODELS, offsets)):
        model_dict = class_data.get(model["name"], {})
        ys = [model_dict.get(c, {}).get("recall", float("nan")) for c in clusters]
        ys_plot = [0.0 if np.isnan(v) else v for v in ys]

        ax.bar(
            x + offset, ys_plot,
            width=bar_w * 0.92,
            color=model["color"],
            label=model["label"],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        for j, v in enumerate(ys):
            if not np.isnan(v) and v > 0.01:
                ax.text(
                    x[j] + offset, v + 0.016,
                    f"{v:.2f}",
                    ha="center", va="bottom", fontsize=6.5,
                    color=plot_style.DARK,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(clusters, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Recall")
    ax.set_title("Touch Recall by Object Class  (Epic Kitchen)", fontsize=12, fontweight="bold")
    ax.set_xlim(-0.7, n_c - 0.3)
    ax.legend(loc="upper right", fontsize=9)

    plt.savefig(output)
    plt.close(fig)
    print(f"Saved → {output}")


# ── Plot 3: Touch zone heatmaps (one figure per dataset) ─────────────────────

def plot_zone_dataset(
    model_grids: dict[str, tuple[np.ndarray, np.ndarray] | None],
    dataset_label: str,
    output: Path,
    grid: int = 8,
) -> None:
    """
    model_grids: {model_name → (recall_grid, count_grid) | None}
    Layout: 1 row × 3 cols (one per model).
    Colorbar vmin is anchored at the minimum recall across all three models.
    """
    plot_style.apply()

    model_names  = [m["name"]  for m in _MODELS]
    model_labels = [m["label"] for m in _MODELS]
    n_cols = len(_MODELS)

    # vmin = min recall across all models for this dataset
    all_vals = []
    for mname in model_names:
        entry = model_grids.get(mname)
        if entry is not None:
            recall, _ = entry
            valid = recall[~np.isnan(recall)]
            if valid.size > 0:
                all_vals.extend(valid.tolist())

    vmin = float(min(all_vals)) if all_vals else 0.0
    vmax = 1.0

    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="#EBEBEB")
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    cell_size = 4.2
    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(cell_size * n_cols + 0.9, cell_size + 0.6),
        squeeze=False,
    )

    for col, (mname, mlabel) in enumerate(zip(model_names, model_labels)):
        ax = axes[0, col]
        ax.grid(False)
        ax.set_title(mlabel, fontsize=11, fontweight="bold")

        entry = model_grids.get(mname)
        if entry is None:
            ax.set_facecolor("#F3F4F6")
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color=plot_style.GRAY)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        recall, count = entry
        masked = np.ma.masked_invalid(recall)
        ax.imshow(masked, cmap=cmap, norm=norm, aspect="equal")

        for cy in range(grid):
            for cx in range(grid):
                n = count[cy, cx]
                if n == 0:
                    ax.text(cx, cy, "–", ha="center", va="center",
                            fontsize=6, color="#BBBBBB")
                else:
                    v = recall[cy, cx]
                    brightness = (v - vmin) / max(1e-9, vmax - vmin)
                    fg = "white" if brightness < 0.25 or brightness > 0.78 else plot_style.DARK
                    ax.text(cx, cy, f"{v:.2f}", ha="center", va="center",
                            fontsize=5.5, color=fg)

        for i in range(grid + 1):
            ax.axhline(i - 0.5, color="white", linewidth=0.5)
            ax.axvline(i - 0.5, color="white", linewidth=0.5)

        ax.set_xticks([])
        ax.set_yticks([])

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, pad=0.02, fraction=0.025, aspect=25)
    cbar.set_label("Recall (hit rate)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_linewidth(0.5)

    fig.suptitle(f"Touch Zone Recall  —  {dataset_label}",
                 fontsize=12, fontweight="bold")
    plt.savefig(output)
    plt.close(fig)
    print(f"Saved → {output}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=_REPO / "results" / "comparison",
        help="Directory to save the three figure files (default: results/comparison/)",
    )
    parser.add_argument(
        "--ek-annotations", type=Path, default=None, metavar="PATH",
        help="Override annotation JSON for Epic Kitchen",
    )
    parser.add_argument(
        "--gh-annotations", type=Path, default=None, metavar="PATH",
        help="Override annotation JSON for Greatest Hits",
    )
    parser.add_argument(
        "--kubric-annotations", type=Path, default=None, metavar="PATH",
        help="Override annotation JSON for Kubric",
    )
    parser.add_argument(
        "--grid", type=int, default=8, metavar="N",
        help="Touch zone grid size (must match annotate_touch_zones.py; default: 8)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    ann_overrides = {}
    if args.ek_annotations:
        ann_overrides["epic_kitchen"] = str(args.ek_annotations)
    if args.gh_annotations:
        ann_overrides["greatest_hits"] = str(args.gh_annotations)
    if args.kubric_annotations:
        ann_overrides["kubric"] = str(args.kubric_annotations)

    # ── Load all dataframes ───────────────────────────────────────────────────

    # dfs[model_name][dataset_name] = DataFrame | None
    dfs: dict[str, dict[str, pd.DataFrame | None]] = {}
    for model in _MODELS:
        dfs[model["name"]] = {}
        for ds in _DATASETS:
            print(f"\nLoading {model['label']} / {ds['label']} …")
            ann_paths = _resolve_annotations(ds, ann_overrides)
            df = load_df(model["csvs"][ds["name"]], ann_paths)
            dfs[model["name"]][ds["name"]] = df

    # ── Compute depth stats ───────────────────────────────────────────────────
    # depth_data[ds_name][model_name] = {bin_label: recall}
    depth_data: dict[str, dict[str, dict[str, float]]] = {
        ds["name"]: {} for ds in _DATASETS
    }
    for model in _MODELS:
        for ds in _DATASETS:
            df = dfs[model["name"]][ds["name"]]
            if df is not None:
                depth_data[ds["name"]][model["name"]] = depth_recall(df)

    # ── Compute cluster stats (EK only) ───────────────────────────────────────
    # class_data[model_name] = {cluster: {"n": int, "recall": float}}
    class_data: dict[str, dict[str, dict]] = {}
    ek_ds = next(ds for ds in _DATASETS if ds["name"] == "epic_kitchen")
    ek_clusters_path = _resolve_clusters(ek_ds)
    if ek_clusters_path:
        with open(ek_clusters_path) as f:
            cluster_map: dict[str, str] = json.load(f)
        for model in _MODELS:
            df = dfs[model["name"]]["epic_kitchen"]
            if df is not None:
                class_data[model["name"]] = cluster_recall(df, cluster_map)
    else:
        print("\n[WARN] Epic Kitchen cluster file not found — skipping class comparison")

    # ── Compute zone grids ────────────────────────────────────────────────────
    # zone_data[model_name][ds_name] = (recall_grid, count_grid)
    zone_data: dict[str, dict[str, tuple[np.ndarray, np.ndarray] | None]] = {}
    for model in _MODELS:
        zone_data[model["name"]] = {}
        for ds in _DATASETS:
            df = dfs[model["name"]][ds["name"]]
            if df is not None:
                zone_data[model["name"]][ds["name"]] = zone_recall(df, args.grid)
            else:
                zone_data[model["name"]][ds["name"]] = None

    # ── Generate figures ──────────────────────────────────────────────────────
    print("\n── Generating figures ──────────────────────────────────────────")

    for ds in _DATASETS:
        slug  = ds["name"]
        label = ds["label"]

        plot_depth_dataset(
            depth_data[slug],
            label,
            args.output_dir / f"depth_{slug}.png",
        )
        plot_zone_dataset(
            {m["name"]: zone_data[m["name"]][slug] for m in _MODELS},
            label,
            args.output_dir / f"zone_{slug}.png",
            grid=args.grid,
        )

    plot_class_comparison(
        class_data,
        args.output_dir / "class_comparison.png",
    )

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
