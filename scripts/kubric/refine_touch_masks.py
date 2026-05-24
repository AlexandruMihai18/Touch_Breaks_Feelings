#!/usr/bin/env python3
"""
refine_touch_masks.py — compute GT-depth and predicted-depth refined touch
masks for Kubric, enabling a direct comparison of how depth quality affects
touch detection performance.

For every touch-annotated frame, three files are written (originals untouched):
  <stem>_touch_refined_gt.png   — touch mask refined with GT depth
                                   → annotation field: touch_refined_gt_path
  <stem>_touch_refined_pred.png — touch mask refined with Depth-Anything-V2
                                   → annotation field: touch_refined_pred_path
  <stem>_depth_pred.png         — predicted depth uint16 PNG
                                   → annotation field: depth_pred_path

The GT depth at depth_path and the original touch mask at target_path are
never modified.  No-touch entries are skipped (they carry no object masks).

Both depth inputs are normalised to [0, 1] so that abs_d_threshold carries
the same meaning for both variants.

Parallel mode (--num-jobs / --job-index)
-----------------------------------------
Each SLURM array task processes a round-robin shard of touch entries and writes
mask files independently.  JSON write-back is skipped in parallel mode to avoid
concurrent overwrites.  After all tasks finish, run this script once in
single-job mode (default) to update the annotation JSON:

    python scripts/kubric/refine_touch_masks.py \\
        --anno-dir /scratch-shared/$USER/kubric_movi_a_256/annotations \\
        --splits val

Usage
-----
    python scripts/kubric/refine_touch_masks.py
    python scripts/kubric/refine_touch_masks.py --splits val --overwrite
    python scripts/kubric/refine_touch_masks.py --dry-run
    # parallel (called from SLURM array):
    python scripts/kubric/refine_touch_masks.py --num-jobs 8 --job-index 3
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from touch_detection_alg.pipeline import compute_depth
from touch_detection_alg.touch import compute_touch_region_v2

_ANNO_DIR = _PROJECT_ROOT / "data" / "kubric_movi_a_256" / "annotations"


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute GT-depth and predicted-depth refined touch masks for Kubric.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--anno-dir", default=str(_ANNO_DIR),
        help="Directory containing split annotation JSON files.",
    )
    p.add_argument(
        "--splits", nargs="+", default=["val"], metavar="SPLIT",
        help="Annotation splits to process.",
    )
    p.add_argument("--dilation-radius", type=int, default=10,
                   help="Contact zone dilation radius (px).")
    p.add_argument("--abs-d-threshold", type=float, default=0.05,
                   help="Max absolute normalised depth gap [0,1] to keep a contact pixel.")
    p.add_argument("--local-radius", type=int, default=10,
                   help="Box-window half-width (px) for per-pixel local depth estimates.")
    p.add_argument(
        "--guided-filter", action=argparse.BooleanOptionalAction, default=True,
        help="Apply guided image filter to predicted depth before computing touch (default: on).",
    )
    p.add_argument("--refine-radius", type=int, default=4,
                   help="Guided-filter spatial radius (px).")
    p.add_argument("--refine-eps", type=float, default=0.1,
                   help="Guided-filter regularisation ε.")
    p.add_argument(
        "--num-jobs", type=int, default=1,
        help="Total number of parallel SLURM array tasks.",
    )
    p.add_argument(
        "--job-index", type=int, default=0,
        help="0-based index of this task (SLURM_ARRAY_TASK_ID).",
    )
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing output files.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would be written without writing anything.")
    return p.parse_args()


# ── Path helpers ──────────────────────────────────────────────────────────────

def _refined_gt_path(target_path: str) -> Path:
    p = Path(target_path)
    return p.parent / (p.stem + "_refined_gt" + p.suffix)


def _refined_pred_path(target_path: str) -> Path:
    p = Path(target_path)
    return p.parent / (p.stem + "_refined_pred" + p.suffix)


def _depth_pred_path(target_path: str) -> Path:
    """Predicted depth lives next to the touch mask as *_depth_pred.png.

    Kubric already has GT depth at depth_path (*_depth.png).  The distinct
    suffix avoids any collision with it.
    """
    p = Path(target_path)
    stem = p.stem.removesuffix("_touch")
    return p.parent / (stem + "_depth_pred" + p.suffix)


def _valid(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _load_gt_depth(depth_path: str) -> np.ndarray:
    """Load Kubric GT depth PNG and normalise to [0, 1]."""
    arr = np.array(Image.open(depth_path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.float32) / 65535.0


def _compute_refined(obj1: np.ndarray, obj2: np.ndarray,
                     depth: np.ndarray, touch_arr: np.ndarray,
                     args) -> np.ndarray:
    """Return refined touch mask, or zeros if the original touch is empty."""
    if not np.any(touch_arr > 0):
        return np.zeros_like(touch_arr)
    return compute_touch_region_v2(
        obj1, obj2, depth,
        dilation_radius=args.dilation_radius,
        abs_d_threshold=args.abs_d_threshold,
        local_radius=args.local_radius,
    )


# ── Per-entry processing ──────────────────────────────────────────────────────

def _process_entry(entry: dict, args) -> str:
    """Compute GT-depth and predicted-depth refined masks for one touch entry.

    Returns 'ok', 'skip', or 'error'.
    Populates entry['touch_refined_gt_path'], entry['touch_refined_pred_path'],
    and entry['depth_pred_path'] as a side-effect (used on JSON write-back).
    """
    target_src   = entry.get("target_path", "")
    gt_depth_src = entry.get("depth_path", "")
    if not target_src:
        return "error"

    out_gt        = _refined_gt_path(target_src)
    out_pred      = _refined_pred_path(target_src)
    out_depth_pred = _depth_pred_path(target_src)

    entry["touch_refined_gt_path"]   = str(out_gt)
    entry["touch_refined_pred_path"] = str(out_pred)
    entry["depth_pred_path"]         = str(out_depth_pred)

    all_done = _valid(out_gt) and _valid(out_pred) and _valid(out_depth_pred)
    if all_done and not args.overwrite:
        return "skip"

    if args.dry_run:
        return "ok"

    # ── Load shared inputs ────────────────────────────────────────────────────
    try:
        img = np.array(Image.open(entry["image_path"]).convert("RGB"))
    except Exception as exc:
        print(f"\n  Load error {Path(entry['image_path']).name}: {exc}")
        return "error"

    try:
        touch_arr = np.array(Image.open(target_src).convert("L"))
        obj1      = np.array(Image.open(entry["object1_mask_path"]).convert("L"))
        obj2      = np.array(Image.open(entry["object2_mask_path"]).convert("L"))
    except Exception as exc:
        print(f"\n  Mask load error {Path(target_src).name}: {exc}")
        return "error"

    # ── Predicted depth (Depth-Anything-V2) ──────────────────────────────────
    try:
        pred_depth = compute_depth(
            img,
            guided_filter=args.guided_filter,
            refine_radius=args.refine_radius,
            refine_eps=args.refine_eps,
        )
        if not _valid(out_depth_pred) or args.overwrite:
            Image.fromarray((pred_depth * 65535).astype(np.uint16)).save(out_depth_pred)
    except Exception as exc:
        print(f"\n  Depth-Anything error {Path(entry['image_path']).name}: {exc}")
        return "error"

    # ── GT-depth refined mask ─────────────────────────────────────────────────
    if not _valid(out_gt) or args.overwrite:
        if not gt_depth_src or not Path(gt_depth_src).exists():
            print(f"\n  [WARN] GT depth missing for {Path(target_src).name} — skipping GT variant")
        else:
            try:
                gt_depth = _load_gt_depth(gt_depth_src)
                refined  = _compute_refined(obj1, obj2, gt_depth, touch_arr, args)
                Image.fromarray(refined).save(out_gt)
            except Exception as exc:
                print(f"\n  GT refinement error {Path(target_src).name}: {exc}")
                return "error"

    # ── Predicted-depth refined mask ──────────────────────────────────────────
    if not _valid(out_pred) or args.overwrite:
        try:
            refined = _compute_refined(obj1, obj2, pred_depth, touch_arr, args)
            Image.fromarray(refined).save(out_pred)
        except Exception as exc:
            print(f"\n  Pred refinement error {Path(target_src).name}: {exc}")
            return "error"

    return "ok"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    anno_dir = Path(args.anno_dir)
    parallel = args.num_jobs > 1

    print("=" * 60)
    print("  Kubric touch mask refinement — GT depth vs. predicted depth")
    print(f"  anno dir       : {anno_dir}")
    print(f"  splits         : {args.splits}")
    print(f"  dilation       : {args.dilation_radius} px")
    print(f"  abs-d-thresh   : {args.abs_d_threshold}")
    print(f"  local radius   : {args.local_radius} px")
    print(f"  guided filter  : {args.guided_filter}"
          + (f"  (r={args.refine_radius}, ε={args.refine_eps})" if args.guided_filter else ""))
    if parallel:
        print(f"  job            : {args.job_index + 1}/{args.num_jobs}")
    print(f"  overwrite      : {args.overwrite}")
    print(f"  dry run        : {args.dry_run}")
    print("=" * 60)

    any_found = False
    for split in args.splits:
        path = anno_dir / f"{split}.json"
        if not path.exists():
            print(f"\n  Not found: {path} — skipping")
            continue

        entries = json.loads(path.read_text())
        any_found = True

        touch = [(i, e) for i, e in enumerate(entries) if e.get("target_path")]
        shard = touch[args.job_index::args.num_jobs]

        print(
            f"\n  [{split}]  {len(entries):,} entries  |  "
            f"{len(touch):,} touch  |  {len(shard):,} this shard"
        )

        if not shard:
            print("  Nothing to process.")
            continue

        if not args.dry_run:
            print("  Loading depth model…")

        ok = skip = errors = 0
        total = len(shard)

        for i, (_, entry) in enumerate(shard, 1):
            result = _process_entry(entry, args)
            if result == "ok":
                ok += 1
            elif result == "skip":
                skip += 1
            else:
                errors += 1
            if i % 10 == 0 or i == total:
                print(f"  {i}/{total}  ok={ok}  skip={skip}  err={errors}",
                      end="\r", flush=True)

        print()
        print(f"  ok={ok}  skip={skip}  errors={errors}")

        # JSON write-back skipped in parallel mode — concurrent writes would
        # overwrite each other.  Run once in single-job mode afterwards to
        # flush touch_refined_gt_path, touch_refined_pred_path, depth_pred_path.
        if not parallel and not args.dry_run:
            path.write_text(json.dumps(entries, indent=2))
            print(
                f"  Updated {path.name} with "
                "touch_refined_gt_path, touch_refined_pred_path, depth_pred_path"
            )

    if not any_found:
        print("No annotation files found — nothing to do.")
        return

    print("\n" + "=" * 60)
    if args.dry_run:
        print("  [DRY RUN] No files written.")
    print("Done.")


if __name__ == "__main__":
    main()
