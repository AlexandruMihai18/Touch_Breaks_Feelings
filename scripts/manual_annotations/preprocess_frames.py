"""
Extract N frames pseudo-uniformly from each video in a directory.

Design goals
------------
- Fast: uses cap.grab() (no decode) for sequential skipping, and
  cap.set() only when the gap to the next target frame is large.
- Resumable: --skip_existing skips any frame already on disk.
- Parallelisable at two levels:
    1. Intra-node: --num_workers spawns a process pool across videos.
       Defaults to the number of CPUs visible to this process
       (respects SLURM's --cpus-per-task via os.sched_getaffinity).
    2. Inter-node (SLURM job array): --job_id / --num_jobs shards the
       full video list so each array task handles a disjoint slice.

Usage — single machine
-----------------------
    python preprocess_frames.py \\
        --video_dir /data/videos \\
        --output_dir /data/frames \\
        --n_frames 20 \\
        --num_workers 16

Usage — SLURM job array
------------------------
    #SBATCH --array=0-7
    python preprocess_frames.py \\
        --video_dir /data/videos \\
        --output_dir /data/frames \\
        --n_frames 20 \\
        --num_jobs  $SLURM_ARRAY_TASK_COUNT \\
        --job_id    $SLURM_ARRAY_TASK_ID

Output layout
-------------
    output_dir/
        video_stem_1/
            frame_000042.jpg
            frame_000213.jpg
            ...
        video_stem_2/
            ...
    extraction_log.jsonl   <- one JSON line per video; append-safe
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parents[2]))

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".flv", ".wmv"}

# If the gap to the next target frame is within this many frames, skip forward
# with grab() (no decode) rather than seeking. Tune based on your typical GOP size.
SEQUENTIAL_THRESHOLD = 60


def sample_frame_indices(
    total_frames: int, n_frames: int, seed: int | None = None
) -> list[int]:
    """Return n_frames pseudo-uniformly spaced frame indices."""
    rng = np.random.default_rng(seed)

    if total_frames <= n_frames:
        return list(range(total_frames))

    segment = total_frames // n_frames
    indices = []
    for i in range(n_frames):
        start = i * segment
        end = min(start + segment - 1, total_frames - 1)
        indices.append(int(rng.integers(start, end + 1)))

    return sorted(indices)


# =========================
# Single-video extraction
# =========================
def _extract_one(
    video_path: Path,
    output_dir: Path,
    n_frames: int,
    image_quality: int,
    seed: int | None,
    skip_existing: bool,
) -> dict:
    """Extract frames from a single video.  Called inside worker processes.

    Returns a result dict written to the log.
    """
    t0 = time.perf_counter()
    video_name = video_path.stem
    out_dir = output_dir / video_name
    result = {"video": str(video_path), "frames_saved": 0, "error": None}

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            result["error"] = "Could not open video"
            return result

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 1:
            result["error"] = "Empty video (0 frames reported)"
            cap.release()
            return result

        indices = sample_frame_indices(total, n_frames, seed=seed)
        out_dir.mkdir(parents=True, exist_ok=True)

        frames_saved = _write_frames(
            cap, indices, out_dir, image_quality, skip_existing
        )
        cap.release()

        result["frames_saved"] = frames_saved
        result["total_frames"] = total
        result["elapsed_s"] = round(time.perf_counter() - t0, 3)

    except Exception as exc:
        result["error"] = str(exc)

    return result


def _write_frames(
    cap: cv2.VideoCapture,
    indices: list[int],
    out_dir: Path,
    image_quality: int,
    skip_existing: bool,
) -> int:
    """Write each target frame to disk using the seek-or-sequential strategy.

    For gaps <= SEQUENTIAL_THRESHOLD frames we advance with cap.grab()
    (no pixel decode) rather than seeking, which is significantly faster
    on compressed codecs because it avoids repeated GOP lookups.
    """
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, image_quality]
    frames_saved = 0
    current_pos = -1  # track where the capture head is

    for idx in indices:
        out_path = out_dir / f"frame_{idx:06d}.jpg"

        if skip_existing and out_path.exists():
            frames_saved += 1
            current_pos = idx
            continue

        gap = idx - current_pos
        if current_pos < 0 or gap > SEQUENTIAL_THRESHOLD:
            # Seeking is cheaper than grabbing across a large gap
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            current_pos = idx
        else:
            # grab() decodes nothing — just advances the demuxer position
            while current_pos < idx:
                cap.grab()
                current_pos += 1

        ret, frame = cap.read()
        if ret:
            cv2.imwrite(str(out_path), frame, encode_params)
            frames_saved += 1
            current_pos += 1

    return frames_saved


# =========================
# Batch runner
# =========================
def collect_videos(video_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in video_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def shard(items: list, num_jobs: int, job_id: int) -> list:
    """Return the slice of items assigned to this SLURM array task."""
    return items[job_id::num_jobs]


def cpu_count() -> int:
    """Number of CPUs available to this process (respects SLURM affinity)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def run(args: argparse.Namespace) -> None:
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_videos = collect_videos(video_dir)
    if not all_videos:
        print(f"No videos found in {video_dir}")
        return

    # Optionally skip videos whose output directory already contains frames
    if args.skip_processed_videos:
        unprocessed = [
            v
            for v in all_videos
            if not any((output_dir / v.stem).glob("*.jpg"))
        ]
        skipped = len(all_videos) - len(unprocessed)
        if skipped:
            print(f"Skipping {skipped} already-processed video(s).")
        all_videos = unprocessed

    # Optionally shard across SLURM array tasks
    videos = (
        shard(all_videos, args.num_jobs, args.job_id)
        if args.num_jobs > 1
        else all_videos
    )
    print(
        f"Processing {len(videos)} / {len(all_videos)} videos  "
        f"(job {args.job_id}/{args.num_jobs}  ·  {args.num_workers} workers)"
    )

    log_path = output_dir / f"extraction_log_job{args.job_id}.jsonl"

    num_workers = args.num_workers or cpu_count()

    with (
        ProcessPoolExecutor(max_workers=num_workers) as pool,
        open(log_path, "a") as log_f,
        tqdm(total=len(videos), unit="video", dynamic_ncols=True) as pbar,
    ):
        futures = {
            pool.submit(
                _extract_one,
                video_path=v,
                output_dir=output_dir,
                n_frames=args.n_frames,
                image_quality=args.image_quality,
                seed=args.seed,
                skip_existing=args.skip_existing,
            ): v
            for v in videos
        }

        errors = 0
        for future in as_completed(futures):
            result = future.result()
            log_f.write(json.dumps(result) + "\n")
            log_f.flush()

            if result["error"]:
                errors += 1
                pbar.set_postfix(errors=errors)
            pbar.update(1)

    print(f"Done. {errors} errors. Log: {log_path}")


# =========================
# CLI
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract frames from videos for ML preprocessing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--video_dir",
        required=True,
        help="Root directory to search for videos (recursive)",
    )
    p.add_argument(
        "--output_dir", required=True, help="Where to write extracted frames"
    )
    p.add_argument(
        "--n_frames", type=int, default=20, help="Frames to extract per video"
    )
    p.add_argument("--image_quality", type=int, default=95, help="JPEG quality (1-100)")
    p.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Worker processes (default: all CPUs visible to this process)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for reproducible frame sampling",
    )
    p.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip frames already on disk (enables resuming interrupted runs)",
    )
    p.add_argument(
        "--no_skip_existing",
        dest="skip_existing",
        action="store_false",
        help="Re-extract and overwrite existing frames",
    )
    p.add_argument(
        "--skip_processed_videos",
        action="store_true",
        default=False,
        help="Skip any video whose output directory already contains at least one frame",
    )
    # SLURM job array sharding
    p.add_argument(
        "--num_jobs",
        type=int,
        default=1,
        help="Total number of SLURM array tasks (set to $SLURM_ARRAY_TASK_COUNT)",
    )
    p.add_argument(
        "--job_id",
        type=int,
        default=0,
        help="Index of this task (set to $SLURM_ARRAY_TASK_ID, 0-indexed)",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
