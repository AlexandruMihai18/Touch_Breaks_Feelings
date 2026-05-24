"""Merge binary and point prediction CSVs per experiment and write to results/.

Binary CSVs are expected to have: <image_path_col>, label, pred
Point  CSVs are expected to have: <image_path_col>, touch_x, touch_y

Output columns: frame_path, label, prediction, x_touch, y_touch
  - pred     -> prediction   (to match the evaluation pipeline)
  - touch_x  -> x_touch      (to match the evaluation pipeline)
  - touch_y  -> y_touch      (to match the evaluation pipeline)

Experiments with no point counterpart produce a binary-only CSV (no x_touch/y_touch).
Merge is an inner join on the detected image-path column.

Note: directory names containing "mlp" without a trailing digit were intended as
"mlp2" — the output names reflect the corrected label (mlp2).

Usage
-----
    # Merge everything
    python scripts/evaluation/merge_predictions.py

    # Preview without writing
    python scripts/evaluation/merge_predictions.py --dry-run

    # Merge a subset by run name
    python scripts/evaluation/merge_predictions.py --runs ek_dino_mlp2 gh_dino_linear
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]

# ── Merge registry ────────────────────────────────────────────────────────────
# binary / point paths are relative to _REPO.
# point=None means binary-only (no point counterpart exists).

_MERGES: list[dict] = [

    # ── Epic Kitchen — DINO ───────────────────────────────────────────────────
    dict(
        run_name = "ek_dino_mlp2",
        dataset  = "epic_kitchen",
        binary   = "checkpoints/dino_epic_binary_mlp/epic_kitchen_dino_binary_mlp_prediction.csv",
        point    = "checkpoints/dino_epic_point_mlp2/epic_kitchen_dino_point_mlp2_prediction.csv",
        output   = "results/ek_dino_mlp2_prediction.csv",
    ),
    dict(
        run_name = "ek_dino_mlp3",
        dataset  = "epic_kitchen",
        binary   = "checkpoints/dino_epic_binary_mlp3/epic_kitchen_dino_binary_mlp3_prediction.csv",
        point    = "checkpoints/dino_epic_point_mlp3/epic_kitchen_dino_point_mlp3_prediction.csv",
        output   = "results/ek_dino_mlp3_prediction.csv",
    ),
    dict(
        run_name = "ek_dino_mlp4",
        dataset  = "epic_kitchen",
        binary   = "checkpoints/dino_epic_binary_mlp4/epic_kitchen_dino_binary_mlp4_prediction.csv",
        point    = "checkpoints/dino_epic_point_mlp4/epic_kitchen_dino_point_mlp4_prediction.csv",
        output   = "results/ek_dino_mlp4_prediction.csv",
    ),
    dict(
        run_name = "ek_dino_linear",
        dataset  = "epic_kitchen",
        binary   = "checkpoints/dino_epic_binary_linear/epic_kitchen_dino_binary_linear_prediction.csv",
        point    = "checkpoints/dino_epic_point_linear/epic_kitchen_dino_point_linear_prediction.csv",
        output   = "results/ek_dino_linear_prediction.csv",
    ),

    # ── Greatest Hits — DINO ─────────────────────────────────────────────────
    dict(
        run_name = "gh_dino_mlp2",
        dataset  = "greatest_hits",
        binary   = "checkpoints/dino_greatest_binary_mlp/greatest_hits_dino_binary_mlp2_prediction.csv",
        point    = "checkpoints/dino_greatest_point_mlp/greatest_hits_dino_point_mlp2_prediction.csv",
        output   = "results/gh_dino_mlp2_prediction.csv",
    ),
    dict(
        run_name = "gh_dino_mlp3",
        dataset  = "greatest_hits",
        binary   = "checkpoints/dino_greatest_binary_mlp3/greatest_hits_dino_binary_mlp3_prediction.csv",
        point    = "checkpoints/dino_greatest_point_mlp3/greatest_hits_dino_point_mlp3_prediction.csv",
        output   = "results/gh_dino_mlp3_prediction.csv",
    ),
    dict(
        run_name = "gh_dino_mlp4",
        dataset  = "greatest_hits",
        binary   = "checkpoints/dino_greatest_binary_mlp4/greatest_hits_dino_binary_mlp4_prediction.csv",
        point    = "checkpoints/dino_greatest_point_mlp4/greatest_hits_dino_point_mlp4_prediction.csv",
        output   = "results/gh_dino_mlp4_prediction.csv",
    ),
    dict(
        run_name = "gh_dino_linear",
        dataset  = "greatest_hits",
        binary   = "checkpoints/dino_greatest_binary_linear/greatest_hits_dino_binary_linear_prediction.csv",
        point    = "checkpoints/dino_greatest_point_linear/greatest_hits_dino_point_linear_prediction.csv",
        output   = "results/gh_dino_linear_prediction.csv",
    ),

    # ── Kubric — DINO ─────────────────────────────────────────────────────────
    dict(
        run_name = "kubric_dino_mlp2",
        dataset  = "kubric",
        binary   = "checkpoints/dino_kubric_binary_mlp2/kubric_dino_binary_mlp2_prediction.csv",
        point    = "checkpoints/dino_kubric_point_mlp2/kubric_dino_point_mlp2_prediction.csv",
        output   = "results/kubric_dino_mlp2_prediction.csv",
    ),
    dict(
        run_name = "kubric_dino_mlp3",
        dataset  = "kubric",
        binary   = "checkpoints/dino_kubric_binary_mlp3/kubric_dino_binary_mlp3_prediction.csv",
        point    = "checkpoints/dino_kubric_point_mlp3/kubric_dino_point_mlp3_prediction.csv",
        output   = "results/kubric_dino_mlp3_prediction.csv",
    ),
    dict(
        run_name = "kubric_dino_mlp4",
        dataset  = "kubric",
        binary   = "checkpoints/dino_kubric_binary_mlp4/kubric_dino_binary_mlp4_prediction.csv",
        point    = "checkpoints/dino_kubric_point_mlp4/kubric_dino_point_mlp4_prediction.csv",
        output   = "results/kubric_dino_mlp4_prediction.csv",
    ),
    dict(
        run_name = "kubric_dino_linear",
        dataset  = "kubric",
        binary   = "checkpoints/dino_kubric_binary_linear/kubric_dino_binary_linear_prediction.csv",
        point    = "checkpoints/dino_kubric_point_linear/kubric_dino_point_linear_prediction.csv",
        output   = "results/kubric_dino_linear_prediction.csv",
    ),

    # ── Greatest Hits — Qwen (binary + point finetuned separately) ────────────
    dict(
        run_name = "gh_qwen_ft",
        dataset  = "greatest_hits",
        binary   = "checkpoints/qwen_greatest_binary/final_adapter/greatest_hits_ft_qwen_binary_prediction.csv",
        point    = "checkpoints/qwen_greatest_point/final_adapter/greatest_hits_ft_qwen_point_prediction.csv",
        output   = "results/gh_qwen_ft_prediction.csv",
    ),

    # ── Epic Kitchen — Qwen (binary only) ─────────────────────────────────────
    dict(
        run_name = "ek_qwen_ft_binary",
        dataset  = "epic_kitchen",
        binary   = "checkpoints/qwen_epic_binary/final_adapter/epic_kitchen_ft_qwen_binary_prediction.csv",
        point    = None,
        output   = "results/ek_qwen_ft_binary_prediction.csv",
    ),

    # ── Kubric — Qwen binary model ─────────────────────────────────────────────
    dict(
        run_name = "kubric_qwen_ft_binary",
        dataset  = "kubric",
        binary   = "checkpoints/qwen_kubric_binary/final_adapter/kubric_ft_qwen_binary_prediction.csv",
        point    = None,
        output   = "results/kubric_qwen_ft_binary_prediction.csv",
    ),

    # ── Kubric — Qwen point model (evaluated on both tasks) ───────────────────
    dict(
        run_name = "kubric_qwen_ft_point",
        dataset  = "kubric",
        binary   = "checkpoints/qwen_kubric_point/final_adapter/kubric_ft_qwen_binary_prediction.csv",
        point    = "checkpoints/qwen_kubric_point/final_adapter/kubric_ft_qwen_point_prediction.csv",
        output   = "results/kubric_qwen_ft_point_prediction.csv",
    ),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

_PATH_COL_CANDIDATES = ["frame_path", "image_path", "img_path", "path"]


def _detect_path_col(df: pd.DataFrame, src: str) -> str:
    for col in _PATH_COL_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(
        f"No image-path column found in {src}.\n"
        f"  Columns: {list(df.columns)}\n"
        f"  Expected one of: {_PATH_COL_CANDIDATES}"
    )


def _merge_one(spec: dict, dry_run: bool) -> bool:
    binary_path = _REPO / spec["binary"] if spec["binary"] else None
    point_path  = _REPO / spec["point"]  if spec["point"]  else None
    out_path    = _REPO / spec["output"]

    # Existence checks
    missing = []
    if binary_path and not binary_path.exists():
        missing.append(f"binary: {binary_path}")
    if point_path and not point_path.exists():
        missing.append(f"point:  {point_path}")
    if missing:
        for m in missing:
            print(f"  [SKIP] Not found — {m}")
        return False

    if dry_run:
        if binary_path:
            print(f"  binary: {binary_path}")
        if point_path:
            print(f"  point:  {point_path}")
        print(f"  ->      {out_path}")
        return True

    frames: list[pd.DataFrame] = []

    if binary_path:
        bin_df = pd.read_csv(binary_path)
        path_col = _detect_path_col(bin_df, str(binary_path))
        bin_df = (
            bin_df[[path_col, "label", "prediction"]]
            .rename(columns={path_col: "frame_path", "prediction": "prediction"})
        )
        frames.append(("binary", bin_df))

    if point_path:
        pt_df = pd.read_csv(point_path)
        path_col = _detect_path_col(pt_df, str(point_path))
        pt_df = (
            pt_df[[path_col, "x_touch", "y_touch"]]
            .rename(columns={path_col: "frame_path", "x_touch": "x_touch", "y_touch": "y_touch"})
        )
        frames.append(("point", pt_df))

    if len(frames) == 2:
        merged = frames[0][1].merge(frames[1][1], on="frame_path", how="inner")
    else:
        merged = frames[0][1]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"  -> {out_path}  ({len(merged):,} rows)")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--runs", nargs="+", default=None, metavar="RUN_NAME",
        help="Merge only these run names (default: all)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without writing any files",
    )
    args = parser.parse_args()

    merges = _MERGES
    if args.runs:
        merges = [m for m in merges if m["run_name"] in args.runs]

    if not merges:
        print("No matching runs found.")
        return

    print(f"{'─' * 60}")
    print(f"  Merge predictions  —  {len(merges)} run(s)")
    if args.dry_run:
        print("  [DRY RUN]")
    print(f"{'─' * 60}\n")

    passed = failed = 0
    for i, spec in enumerate(merges, 1):
        label = f"[{i}/{len(merges)}]  {spec['run_name']}  ({spec['dataset']})"
        print(label)
        ok = _merge_one(spec, args.dry_run)
        if ok:
            passed += 1
        else:
            failed += 1
        print()

    if not args.dry_run:
        print(f"{'─' * 60}")
        print(f"  Done — merged={passed}  skipped={failed}")
        print(f"{'─' * 60}")


if __name__ == "__main__":
    main()
