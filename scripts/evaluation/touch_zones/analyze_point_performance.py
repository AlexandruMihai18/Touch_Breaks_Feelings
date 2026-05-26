"""Evaluate touch-point regression predictions on a spatial grid.

The input CSV must come from a DINO point-task evaluation run and contain:
    frame_path, x_touch, y_touch  — predicted touch in pixel coords (original image)

Ground-truth grid cells come from annotation JSONs where x_touch / y_touch
are integer cell indices in [0, N) produced by annotate_touch_zones.py.

Distance metric
---------------
All coordinates are normalized to [0, 1] before comparison so the metric is
independent of image resolution and model input size.

Both DINO and Qwen output x_touch / y_touch as pixel coordinates; these are
divided by the original image dimensions (width, height) to normalize.

The GT cell centroid is computed as ((x_gt + 0.5) / grid, (y_gt + 0.5) / grid),
also in [0, 1].  The error per sample is the joint 2-D Euclidean distance
between the normalized predicted point and the GT cell centroid:

    error_norm = ||(x_pred_norm, y_pred_norm) - (cx_norm, cy_norm)||₂

This avoids double-discretization noise and is directly comparable across
models regardless of their input resolution.

Two outputs
-----------
  <stem>_hit_rate.png   — per-GT-cell fraction of predictions landing in the
                          exact correct cell (hit@1)
  <stem>_mean_error.png — per-GT-cell mean normalized error (→ GT centroid,
                          lower = better; range [0, √2])

Usage
-----
    python scripts/evaluation/touch_zones/analyze_point_performance.py \\
        results/predictions_point.csv \\
        --annotations /path/to/val.json \\
        --output results/evaluation/my_run/point_heatmap.png \\
        [--grid 8]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import enrich_df
import plot_style


# ── Image size cache ──────────────────────────────────────────────────────────

_size_cache: dict[str, tuple[int, int]] = {}


def _image_size(frame_path: str) -> tuple[int, int]:
    """Return (width, height) without loading pixels (header-only read)."""
    if frame_path not in _size_cache:
        with Image.open(frame_path) as img:
            _size_cache[frame_path] = img.size
    return _size_cache[frame_path]


# ── Per-row computation ───────────────────────────────────────────────────────

def _augment_row(
    frame_path: str,
    x_pred: float,
    y_pred: float,
    x_gt: int,
    y_gt: int,
    grid: int,
    coord_scale: float | None = None,
    cx_gt_n: float | None = None,
    cy_gt_n: float | None = None,
) -> tuple[int, int, float]:
    """Return (x_pred_cell, y_pred_cell, error_norm).

    coord_scale: if given, divide x_pred/y_pred by this value to normalize
    (e.g. 1000 for Qwen models that output in [0, 1000] scale).
    If None, normalize by the original image dimensions read from disk (DINO).
    """
    if coord_scale is not None:
        x_n = x_pred / coord_scale
        y_n = y_pred / coord_scale
    else:
        w, h = _image_size(frame_path)
        x_n = x_pred / w
        y_n = y_pred / h
    if cx_gt_n is None or cy_gt_n is None:
        cx_gt_n = (x_gt + 0.5) / grid
        cy_gt_n = (y_gt + 0.5) / grid
    xc = max(0, min(int(x_n * grid), grid - 1))
    yc = max(0, min(int(y_n * grid), grid - 1))
    err = float(np.sqrt((x_n - cx_gt_n) ** 2 + (y_n - cy_gt_n) ** 2))
    return xc, yc, err


# ── Per-cell statistics ───────────────────────────────────────────────────────

def _compute_grids(
    df: pd.DataFrame,
    grid: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (hit_rate, mean_error_norm, count) arrays shaped (grid, grid).

    Indexed as [y, x] — row y=0 is the top of the image.
    """
    hit_rate = np.full((grid, grid), np.nan)
    mean_err = np.full((grid, grid), np.nan)
    count    = np.zeros((grid, grid), dtype=int)

    g = df.groupby(["y_gt", "x_gt"])
    for (y, x), grp in g:
        if not (0 <= x < grid and 0 <= y < grid):
            continue
        n = len(grp)
        count[y, x]    = n
        hits           = (grp["x_pred_cell"] == x) & (grp["y_pred_cell"] == y)
        hit_rate[y, x] = hits.sum() / n
        mean_err[y, x] = grp["error_norm"].mean()

    return hit_rate, mean_err, count


# ── Heatmap rendering ─────────────────────────────────────────────────────────

def _plot_heatmap(
    values: np.ndarray,
    count: np.ndarray,
    title: str,
    cbar_label: str,
    output_path: Path,
    cmap_name: str,
    vmin: float,
    vmax: float,
    fmt: str,
) -> None:
    plot_style.apply()
    grid = values.shape[0]
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="#EBEBEB")

    fig, ax = plt.subplots(figsize=(6.0, 5.5))
    masked = np.ma.masked_invalid(values)
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")

    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046, aspect=20)
    cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_linewidth(0.5)

    for y in range(grid):
        for x in range(grid):
            n = count[y, x]
            if n == 0:
                ax.text(x, y, "–", ha="center", va="center",
                        fontsize=8, color="#AAAAAA")
            else:
                v = values[y, x]
                txt = fmt.format(v) + f"\n({n})"
                mid = (vmin + vmax) / 2
                if cmap_name.endswith("_r"):
                    fg = "white" if v > mid * 1.3 else plot_style.DARK
                else:
                    fg = "white" if (v < vmin + (vmax - vmin) * 0.25
                                     or v > vmin + (vmax - vmin) * 0.78) else plot_style.DARK
                ax.text(x, y, txt, ha="center", va="center",
                        fontsize=7.0, color=fg, linespacing=1.35)

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
    ax.set_ylabel("y zone  (top → bottom)", labelpad=6)
    ax.set_title(title, pad=14)
    ax.grid(False)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "csv", type=Path,
        help="DINO point-task predictions CSV (x_touch, y_touch = pixel coords)",
    )
    parser.add_argument(
        "--annotations", nargs="+", type=Path, required=True,
        help="Annotation JSON files with GT x_touch/y_touch cell indices",
    )
    parser.add_argument(
        "--grid", type=int, default=8, metavar="N",
        help="Grid size N — must match annotate_touch_zones.py (default: 8)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Base output path; _hit_rate.png and _mean_error.png are appended "
             "(default: <csv_stem>_point_heatmap next to the CSV)",
    )
    parser.add_argument(
        "--mask-mode", choices=["auto", "zero_coord", "predicted_touch"], default="auto",
        help="Secondary mask applied within label==1 samples for RMSE. "
             "predicted_touch: no further filtering (default for DINO). "
             "zero_coord: additionally exclude rows where x_pred=0 & y_pred=0 "
             "(Qwen sentinel for 'no touch predicted'). "
             "auto (default): uses predicted_touch.",
    )
    parser.add_argument(
        "--coord-scale", type=float, default=None, metavar="SCALE",
        help="Divide x_pred/y_pred by this value to normalize to [0, 1]. "
             "Use 1000 for Qwen models (output in [0, 1000] scale). "
             "Default: normalize by original image dimensions (DINO).",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(args.csv)

    df = pd.read_csv(args.csv)

    # ── Preserve predicted pixel coords before enrich_df overwrites them ──────
    # The binary-task CSV has x_touch=NaN; for the point task they are pixels.
    df["x_pred"] = pd.to_numeric(df.get("x_touch"), errors="coerce")
    df["y_pred"] = pd.to_numeric(df.get("y_touch"), errors="coerce")

    n_with_pred = df["x_pred"].notna().sum()
    if n_with_pred == 0:
        print("[ERROR] No valid x_touch / y_touch values in CSV — "
              "is this a point-task prediction file?")
        sys.exit(1)
    print(f"  Rows with predicted coords : {n_with_pred:,} / {len(df):,}")

    # ── Join GT cell indices from annotation JSONs ────────────────────────────
    df = enrich_df(df, args.annotations)
    df = df.rename(columns={"x_touch": "x_gt", "y_touch": "y_gt"})

    df = df.dropna(subset=["x_pred", "y_pred", "x_gt", "y_gt"])
    df["x_gt"] = df["x_gt"].astype(int)
    df["y_gt"] = df["y_gt"].astype(int)

    if df.empty:
        print("[ERROR] No rows matched between predictions and annotation GT cells. "
              "Check that --annotations uses the same split as the CSV.")
        sys.exit(1)
    print(f"  Rows with GT cell indices  : {len(df):,}")

    # ── Resolve mask mode ─────────────────────────────────────────────────────
    mask_mode = args.mask_mode
    if mask_mode == "auto":
        mask_mode = "predicted_touch"

    # ── Check for mask-centroid annotation (preferred GT over cell centroid) ───
    has_centroid = (
        "x_touch_centroid" in df.columns
        and "y_touch_centroid" in df.columns
        and df["x_touch_centroid"].notna().any()
    )
    gt_source = "mask centroid (x_touch_centroid)" if has_centroid else f"cell centroid ({args.grid}×{args.grid} grid)"
    print(f"  GT source: {gt_source}")

    # ── Map predictions to grid cells and compute normalized error ────────────
    results = []
    skipped = 0
    for _, row in df.iterrows():
        if has_centroid:
            cx = row.get("x_touch_centroid")
            cy = row.get("y_touch_centroid")
            cx_gt_n = float(cx) if cx is not None and pd.notna(cx) else None
            cy_gt_n = float(cy) if cy is not None and pd.notna(cy) else None
        else:
            cx_gt_n = cy_gt_n = None
        try:
            xc, yc, err = _augment_row(
                str(row["frame_path"]),
                float(row["x_pred"]),
                float(row["y_pred"]),
                int(row["x_gt"]),
                int(row["y_gt"]),
                args.grid,
                coord_scale=args.coord_scale,
                cx_gt_n=cx_gt_n,
                cy_gt_n=cy_gt_n,
            )
            results.append((xc, yc, err))
        except Exception:
            results.append((None, None, None))
            skipped += 1

    df["x_pred_cell"] = [r[0] for r in results]
    df["y_pred_cell"] = [r[1] for r in results]
    df["error_norm"]  = [r[2] for r in results]
    df = df.dropna(subset=["x_pred_cell", "y_pred_cell", "error_norm"])
    df["x_pred_cell"] = df["x_pred_cell"].astype(int)
    df["y_pred_cell"] = df["y_pred_cell"].astype(int)

    if skipped:
        print(f"  [WARN] {skipped} rows skipped (image read error or missing frame_path)")

    # ── Summary statistics ────────────────────────────────────────────────────
    n = len(df)
    hit1 = (df["x_pred_cell"] == df["x_gt"]) & (df["y_pred_cell"] == df["y_gt"])

    print(f"\n  Results  (n={n:,},  grid={args.grid}×{args.grid},  coords=normalized [0,1])")
    print(f"    Hit@1  (exact cell match)            : "
          f"{hit1.mean():.3f}  ({hit1.sum()}/{n})")
    print(f"    Mean error  (→ GT centroid)          : {df['error_norm'].mean():.4f}")
    print(f"    Median error                         : {df['error_norm'].median():.4f}")

    # Base: only label==1 (ground truth touch samples).
    # For zero_coord models (Qwen): additionally exclude (0,0) sentinel predictions
    # within label==1 (those are false negatives with no real coordinate output).
    label_pos = pd.to_numeric(df.get("label"), errors="coerce").fillna(-1) == 1
    if mask_mode == "zero_coord":
        rmse_keep = label_pos & ~((df["x_pred"] == 0.0) & (df["y_pred"] == 0.0))
    else:  # predicted_touch — keep all label==1 rows
        rmse_keep = label_pos

    df_rmse = df[rmse_keep]
    rmse = float(np.sqrt((df_rmse["error_norm"] ** 2).mean())) if not df_rmse.empty else float("nan")
    n_masked = int((~rmse_keep).sum())
    mode_label = "zero-coord FN" if mask_mode == "zero_coord" else "label=0"
    print(f"    RMSE ({n_masked} {mode_label} masked)          : {rmse:.4f}")


    # ── Per-cell grids ────────────────────────────────────────────────────────
    hit_rate, mean_err, cell_count = _compute_grids(df, args.grid)

    base = args.output or args.csv.with_name(args.csv.stem + "_point_heatmap")
    base = Path(base)

    stats = {
        "hit_at_1":          float(hit1.mean()),
        "mean_error_norm":   float(df["error_norm"].mean()),
        "median_error_norm": float(df["error_norm"].median()),
        "rmse_norm":         rmse,
        "n":                 n,
        "n_masked":          n_masked,
        "mask_mode":         mask_mode,
        "coord_scale":       args.coord_scale,
        "gt_source":         "mask_centroid" if has_centroid else "cell_centroid",
        "grid":              args.grid,
    }
    stats_path = base.with_name(base.stem + "_stats.json")
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"  Saved → {stats_path}")

    valid_hr = hit_rate[~np.isnan(hit_rate) & (hit_rate > 0)]
    hr_vmin = float(valid_hr.min()) if valid_hr.size > 0 else 0.0
    _plot_heatmap(
        hit_rate, cell_count,
        title="Touch-point regression — hit rate\n(prediction lands in GT cell)",
        cbar_label="Hit@1",
        output_path=base.with_name(base.stem + "_hit_rate.png"),
        cmap_name="RdYlGn",
        vmin=hr_vmin, vmax=1.0,
        fmt="{:.2f}",
    )

    valid_me = mean_err[~np.isnan(mean_err) & (mean_err > 0)]
    me_vmin = float(valid_me.min()) if valid_me.size > 0 else 0.0
    p90 = float(np.nanpercentile(valid_me, 90)) if valid_me.size > 0 else 0.5
    _plot_heatmap(
        mean_err, cell_count,
        title="Touch-point regression — mean error (normalized)\n(predicted point → GT cell centroid)",
        cbar_label="Mean error (norm, [0–√2])",
        output_path=base.with_name(base.stem + "_mean_error.png"),
        cmap_name="RdYlGn_r",
        vmin=me_vmin, vmax=p90,
        fmt="{:.3f}",
    )


if __name__ == "__main__":
    main()
