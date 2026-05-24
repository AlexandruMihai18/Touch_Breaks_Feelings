"""Sample failure cases per sub-category for qualitative error analysis.

For each analysis dimension (depth, object_coverage) this script finds
mispredicted samples per bin, randomly samples up to --n-samples of them,
and copies the frame images into a structured output directory.

Output layout
-------------
<output-dir>/
  depth_failures/
    near_distance/          (Close bin: disparity 170-255)
      sample_1.ext
      sample_2.ext
      metadata.csv
    medium_distance/        (Medium bin: disparity 85-170)
      ...
    far_distance/           (Far bin: disparity 0-85)
      ...
  object_size_failures/
    small/                  (coverage < 0.02)
      ...
    medium_small/           (coverage 0.02-0.07)
      ...
    medium_large/           (coverage 0.07-0.15)
      ...
    large/                  (coverage >= 0.15)
      ...

Usage
-----
    python scripts/evaluation/failure_sampling/sample_failures.py \\
        results/predictions.csv \\
        --annotations /path/to/annotations.json \\
        --output-dir  results/evaluation/my_run \\
        [--n-samples 3] [--seed 42]
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import enrich_df


# ── Bin definitions (must match the main analysis scripts) ────────────────────

_DEPTH_BINS = [
    # (dir_name, display_label, lo_inclusive, hi_exclusive)
    ("near_distance",   "Close",   170, 256),
    ("medium_distance", "Medium",   85, 170),
    ("far_distance",    "Far",       0,  85),
]

_COVERAGE_BINS = [
    ("small",        "Small",        0.00,  0.02),
    ("medium_small", "Medium-small", 0.02,  0.07),
    ("medium_large", "Medium-large", 0.07,  0.15),
    ("large",        "Large",        0.15,  1.01),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _error_type(label: int, pred: int) -> str:
    if label == 1 and pred == 0:
        return "FN"
    if label == 0 and pred == 1:
        return "FP"
    return "correct"


def _sample_bin_failures(
    df: pd.DataFrame,
    col: str,
    lo: float,
    hi: float,
    n: int,
    seed: int,
    touch_only: bool,
) -> pd.DataFrame:
    mask = df[col].notna() & (df[col] >= lo) & (df[col] < hi)
    if touch_only:
        mask &= df["label"] == 1
    group = df[mask]
    failures = group[group["label"].astype(int) != group["prediction"].astype(int)]
    if failures.empty:
        return pd.DataFrame()
    k = min(n, len(failures))
    return failures.sample(k, random_state=seed)


def _save_sample(row: pd.Series, dest_dir: Path, idx: int) -> dict:
    """Copy frame image to dest_dir; return a metadata dict for the row."""
    src = Path(str(row["frame_path"]))
    ext = src.suffix or ".jpg"
    dst_name = f"sample_{idx + 1}{ext}"
    dst = dest_dir / dst_name

    meta: dict = {
        "sample":      dst_name,
        "frame_path":  str(src),
        "label":       int(row["label"]),
        "prediction":  int(row["prediction"]),
        "error_type":  _error_type(int(row["label"]), int(row["prediction"])),
    }
    for col in ("depth_touch", "object_coverage", "object_name"):
        if col in row.index and pd.notna(row[col]):
            meta[col] = row[col]

    if src.exists():
        shutil.copy2(src, dst)
    else:
        meta["missing_source"] = True

    return meta


def _process_bins(
    df: pd.DataFrame,
    bins: list[tuple],
    col: str,
    out_root: Path,
    n_samples: int,
    base_seed: int,
    touch_only: bool,
) -> int:
    out_root.mkdir(parents=True, exist_ok=True)
    total = 0

    for i, (dir_name, display_label, lo, hi) in enumerate(bins):
        seed = base_seed + i
        failures = _sample_bin_failures(df, col, lo, hi, n_samples, seed, touch_only)
        bin_dir = out_root / dir_name
        bin_dir.mkdir(exist_ok=True)

        if failures.empty:
            print(f"  [SKIP] {dir_name} ({display_label}): no failures in this bin")
            continue

        rows = [_save_sample(row, bin_dir, j) for j, (_, row) in enumerate(failures.iterrows())]
        pd.DataFrame(rows).to_csv(bin_dir / "metadata.csv", index=False)
        n = len(rows)
        total += n

        fp = sum(1 for r in rows if r["error_type"] == "FP")
        fn = sum(1 for r in rows if r["error_type"] == "FN")
        detail = f"FP={fp} FN={fn}" if not touch_only else f"FN={fn}"
        print(f"  {dir_name} ({display_label}): {n} sample(s) [{detail}]")

    return total


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("csv", type=Path, help="Predictions CSV")
    parser.add_argument("--annotations", nargs="+", type=Path, required=True,
                        help="Annotation JSON file(s)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Run output directory; depth_failures/ and "
                             "object_size_failures/ are created inside it")
    parser.add_argument("--n-samples", type=int, default=3,
                        help="Max failures to sample per sub-category (default: 3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (default: 42)")
    parser.add_argument("--skip", nargs="*", choices=["depth", "coverage"], default=[],
                        metavar="DIM",
                        help="Dimensions to skip: depth  coverage")
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(args.csv)

    df = pd.read_csv(args.csv)
    df = enrich_df(df, args.annotations)

    skipped = set(args.skip or [])

    # ── Depth failures ────────────────────────────────────────────────────────
    if "depth" not in skipped:
        if "depth_touch" not in df.columns or df["depth_touch"].isna().all():
            print("[SKIP] depth failures — depth_touch not populated")
        else:
            print("\nDepth failures (touch-positive only — failures are FN):")
            total = _process_bins(
                df, _DEPTH_BINS, "depth_touch",
                args.output_dir / "depth_failures",
                args.n_samples, args.seed,
                touch_only=True,
            )
            print(f"  → {total} depth failure sample(s) saved")

    # ── Object size (coverage) failures ──────────────────────────────────────
    if "coverage" not in skipped:
        if "object_coverage" not in df.columns or df["object_coverage"].isna().all():
            print("[SKIP] object_size failures — object_coverage not populated")
        else:
            print("\nObject size failures (all samples — failures include FP and FN):")
            total = _process_bins(
                df, _COVERAGE_BINS, "object_coverage",
                args.output_dir / "object_size_failures",
                args.n_samples, args.seed + 100,
                touch_only=False,
            )
            print(f"  → {total} object_size failure sample(s) saved")


if __name__ == "__main__":
    main()
