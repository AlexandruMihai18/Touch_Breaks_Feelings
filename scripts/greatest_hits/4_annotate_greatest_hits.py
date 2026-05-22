"""
Batch-annotate the Greatest Hits dataset using the per-frame SAM pipeline.

For each video folder in GH_FRAMES_ROOT the script:
  1. Calibrates blur thresholds from the folder's frames.
  2. Per frame: detects the stick via Grounding-DINO + SAM,
     locates the object at the stick tip, estimates depth
     (CLAHE → unsharp-mask → Depth-Anything-V2 → guided filter),
     and computes touch via compute_touch_region_v2.
  3. Touch is clipped to the union of the raw stick and object masks.
  4. Saves masks + per-folder manifest.json and dataset.json.

Output layout:

    data/greatest_hits/masks/
        {video_id}/
            frame_XXXXXX.jpg
            frame_XXXXXX_touch.png
            frame_XXXXXX_object.png
            frame_XXXXXX_stick.png
            frame_XXXXXX_depth.png
            manifest.json
            dataset.json

Usage — annotate all folders
-----------------------------
    python scripts/greatest_hits/annotate_greatest_hits.py

Usage — single video
---------------------
    python scripts/greatest_hits/annotate_greatest_hits.py --video-id 2015-03-28-19-34-13_denoised

Usage — skip already-annotated folders
----------------------------------------
    python scripts/greatest_hits/annotate_greatest_hits.py --skip-existing
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parents[2]))

from inference_script.shared import GH_ANNO_DIR, GH_FRAMES_ROOT, GH_MASKS_ROOT
from touch_detection_alg.depth import calibrate_blur_thresholds
from touch_detection_alg.greatest_hits import annotate_folder


def _is_annotated(masks_root: Path, video_id: str) -> bool:
    return (masks_root / video_id / "dataset.json").exists()


def _load_folder(folder: Path) -> tuple[list[np.ndarray], list[Path]]:
    paths = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
    frames = [np.array(Image.open(p).convert("RGB")) for p in paths]
    return frames, paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-annotate Greatest Hits folders with the per-frame SAM pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--frames-dir", type=Path, default=GH_FRAMES_ROOT,
        help="Root folder containing per-video frame sub-directories.",
    )
    parser.add_argument(
        "--masks-dir", type=Path, default=GH_MASKS_ROOT,
        help="Root folder where masks and metadata will be saved.",
    )
    parser.add_argument(
        "--video-id", type=str, default=None,
        help="Annotate only this video folder; omit to process all folders.",
    )
    parser.add_argument(
        "--annotations-dir", type=Path, default=GH_ANNO_DIR,
        help="Directory containing train.json and val.json (built by 3_generate_gh_gt_annotations.py).",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip folders that already have a dataset.json in the masks directory.",
    )
    parser.add_argument(
        "--num-jobs", type=int, default=1,
        help="Total number of parallel SLURM array tasks.",
    )
    parser.add_argument(
        "--job-index", type=int, default=0,
        help="0-based index of this task (SLURM_ARRAY_TASK_ID).",
    )
    # ── Touch region ────────────────────────────────────────────────────────
    parser.add_argument(
        "--dilation", type=int, default=10,
        help="Contact zone dilation radius (px).",
    )
    parser.add_argument(
        "--abs-d-threshold", type=float, default=0.05,
        help="Max absolute depth gap [0,1] to keep a contact pixel.",
    )
    parser.add_argument(
        "--local-radius", type=int, default=10,
        help="Box-window half-width (px) for per-pixel local depth estimates.",
    )
    # ── Depth preprocessing ─────────────────────────────────────────────────
    parser.add_argument(
        "--clahe-clip", type=float, default=2.0,
        help="CLAHE clip limit for contrast enhancement before depth estimation. 0 = off.",
    )
    parser.add_argument(
        "--guided-filter", action=argparse.BooleanOptionalAction, default=True,
        help="Sharpen depth with guided image filter after estimation.",
    )
    parser.add_argument(
        "--refine-radius", type=int, default=4,
        help="Guided-filter spatial radius (px).",
    )
    parser.add_argument(
        "--refine-eps", type=float, default=0.1,
        help="Guided-filter regularisation ε.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.frames_dir.exists():
        print(f"Frames directory not found: {args.frames_dir}")
        sys.exit(1)

    if args.video_id:
        folders = [args.frames_dir / args.video_id]
        if not folders[0].is_dir():
            print(f"Video folder not found: {folders[0]}")
            sys.exit(1)
    else:
        folders = sorted(d for d in args.frames_dir.iterdir() if d.is_dir())

    if not folders:
        print(f"No video folders found in {args.frames_dir}")
        sys.exit(0)

    # Round-robin shard: job i handles indices i, i+num_jobs, i+2*num_jobs, …
    if args.num_jobs > 1:
        folders = folders[args.job_index::args.num_jobs]

    print("=" * 60)
    print("  Greatest Hits auto-annotation")
    print(f"  frames dir       : {args.frames_dir}")
    print(f"  masks dir        : {args.masks_dir}")
    if args.num_jobs > 1:
        print(f"  job              : {args.job_index + 1}/{args.num_jobs}")
    print(f"  videos           : {len(folders)}")
    print(f"  dilation         : {args.dilation} px")
    print(f"  abs-d-threshold  : {args.abs_d_threshold}")
    print(f"  local radius     : {args.local_radius} px")
    print(f"  CLAHE clip       : {args.clahe_clip if args.clahe_clip > 0 else 'off'}")
    print(f"  guided filter    : {args.guided_filter}"
          + (f"  (r={args.refine_radius}, ε={args.refine_eps})" if args.guided_filter else ""))
    print(f"  skip existing    : {args.skip_existing}")
    print("=" * 60)

    skipped = failed = succeeded = 0
    t_start = time.time()

    for folder in tqdm(folders, desc="Annotating", unit="video"):
        video_id = folder.name

        if args.skip_existing and _is_annotated(args.masks_dir, video_id):
            skipped += 1
            tqdm.write(f"[skip]  {video_id}")
            continue

        frames, frame_paths = _load_folder(folder)
        if not frames:
            tqdm.write(f"[empty] {video_id} — no frames found")
            failed += 1
            continue

        calibrate_blur_thresholds(frames)

        try:
            result = annotate_folder(
                folder_name=video_id,
                frames=frames,
                frame_paths=frame_paths,
                masks_root=args.masks_dir,
                annotations_dir=args.annotations_dir,
                dilation=args.dilation,
                abs_d_threshold=args.abs_d_threshold,
                local_radius=args.local_radius,
                guided_filter=args.guided_filter,
                refine_radius=args.refine_radius,
                refine_eps=args.refine_eps,
                clahe_clip=args.clahe_clip,
            )
        except Exception as exc:
            tqdm.write(f"[fail]  {video_id} — {exc}")
            failed += 1
            continue

        first_line = result.splitlines()[0] if result else ""
        if "0/" in first_line:
            tqdm.write(f"[fail]  {video_id} — no frames annotated")
            failed += 1
        else:
            tqdm.write(f"[ok]    {first_line}")
            succeeded += 1

    elapsed = time.time() - t_start
    print(
        f"\nDone in {elapsed:.1f}s — "
        f"{succeeded} annotated, {skipped} skipped, {failed} failed "
        f"(total {len(folders)} folders)"
    )


if __name__ == "__main__":
    main()
