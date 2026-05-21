"""
Extract annotated frames from the Greatest Hits dataset.

For each {timestamp}_denoised.mp4 video, reads the paired {timestamp}_times.txt
file to get the annotated contact moments (in seconds), converts them to frame
indices using the video's FPS, and writes the frames to disk.

Output layout mirrors preprocess_frames.py:

    output_dir/
        {timestamp}_denoised/
            frame_000041.jpg
            frame_000112.jpg
            ...
        extraction_log_job{job_id}.jsonl

Usage — single machine
-----------------------
    python extract_greatest_hits_frames.py \\
        --data_dir  data/greatest_hits/vis-data-256/vis-data-256 \\
        --output_dir data/greatest_hits/frames \\
        --num_workers 8

Usage — SLURM job array
------------------------
    #SBATCH --array=0-7
    python extract_greatest_hits_frames.py \\
        --data_dir  data/greatest_hits/vis-data-256/vis-data-256 \\
        --output_dir data/greatest_hits/frames \\
        --num_jobs  $SLURM_ARRAY_TASK_COUNT \\
        --job_id    $SLURM_ARRAY_TASK_ID
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
from tqdm import tqdm

SEQUENTIAL_THRESHOLD = 60


# =========================
# Times-file parsing
# =========================
def parse_times_file(times_path: Path) -> list[float]:
    """Return annotated timestamps (seconds) from a _times.txt file."""
    timestamps = []
    for line in times_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        t = float(line.split()[0])
        timestamps.append(t)
    return timestamps


# =========================
# Single-video extraction
# =========================
def _extract_one(
    video_path: Path,
    times_path: Path,
    output_dir: Path,
    image_quality: int,
    skip_existing: bool,
) -> dict:
    t0 = time.perf_counter()
    out_dir = output_dir / video_path.stem
    result = {"video": str(video_path), "frames_saved": 0, "error": None}

    try:
        timestamps = parse_times_file(times_path)
        if not timestamps:
            result["error"] = "No timestamps in times file"
            return result

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            result["error"] = "Could not open video"
            return result

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Convert seconds → frame indices; clamp to valid range; deduplicate
        indices = sorted(
            {min(round(t * fps), max(total - 1, 0)) for t in timestamps}
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        frames_saved = _write_frames(cap, indices, out_dir, image_quality, skip_existing)
        cap.release()

        result["frames_saved"] = frames_saved
        result["total_frames"] = total
        result["fps"] = fps
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
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, image_quality]
    frames_saved = 0
    current_pos = -1

    for idx in indices:
        out_path = out_dir / f"frame_{idx:06d}.jpg"

        if skip_existing and out_path.exists():
            frames_saved += 1
            current_pos = idx
            continue

        gap = idx - current_pos
        if current_pos < 0 or gap > SEQUENTIAL_THRESHOLD:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            current_pos = idx
        else:
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
def collect_video_pairs(data_dir: Path) -> list[tuple[Path, Path]]:
    """Return (denoised.mp4, times.txt) pairs that have both files present."""
    pairs = []
    for mp4 in sorted(data_dir.glob("*_denoised.mp4")):
        # Strip _denoised suffix to get the shared timestamp prefix
        prefix = mp4.stem[: -len("_denoised")]
        times = data_dir / f"{prefix}_times.txt"
        if times.exists():
            pairs.append((mp4, times))
    return pairs


def shard(items: list, num_jobs: int, job_id: int) -> list:
    return items[job_id::num_jobs]


def cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def run(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_pairs = collect_video_pairs(data_dir)
    if not all_pairs:
        print(f"No (denoised.mp4, times.txt) pairs found in {data_dir}")
        return

    if args.skip_processed_videos:
        unprocessed = [
            (v, t)
            for v, t in all_pairs
            if not any((output_dir / v.stem).glob("*.jpg"))
        ]
        skipped = len(all_pairs) - len(unprocessed)
        if skipped:
            print(f"Skipping {skipped} already-processed video(s).")
        all_pairs = unprocessed

    pairs = shard(all_pairs, args.num_jobs, args.job_id) if args.num_jobs > 1 else all_pairs

    num_workers = args.num_workers or cpu_count()
    print(
        f"Processing {len(pairs)} / {len(all_pairs)} videos  "
        f"(job {args.job_id}/{args.num_jobs}  ·  {num_workers} workers)"
    )

    log_path = output_dir / f"extraction_log_job{args.job_id}.jsonl"

    with (
        ProcessPoolExecutor(max_workers=num_workers) as pool,
        open(log_path, "a") as log_f,
        tqdm(total=len(pairs), unit="video", dynamic_ncols=True) as pbar,
    ):
        futures = {
            pool.submit(
                _extract_one,
                video_path=v,
                times_path=t,
                output_dir=output_dir,
                image_quality=args.image_quality,
                skip_existing=args.skip_existing,
            ): v
            for v, t in pairs
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
        description="Extract annotated contact frames from the Greatest Hits dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data_dir",
        required=True,
        help="Directory containing *_denoised.mp4 and *_times.txt files",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        help="Root directory to write per-video frame subfolders",
    )
    p.add_argument("--image_quality", type=int, default=95, help="JPEG quality (1-100)")
    p.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Worker processes (default: all CPUs visible to this process)",
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
    p.add_argument(
        "--num_jobs",
        type=int,
        default=1,
        help="Total SLURM array tasks (set to $SLURM_ARRAY_TASK_COUNT)",
    )
    p.add_argument(
        "--job_id",
        type=int,
        default=0,
        help="This task's index (set to $SLURM_ARRAY_TASK_ID, 0-indexed)",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
