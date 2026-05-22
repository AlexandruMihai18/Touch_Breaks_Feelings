"""Standalone Phase 6 re-runner — extract audio for all videos already on disk.

Discovers video IDs from the frames/ directory (populated by phases 1-3), so
it works whether or not annotation JSONs have been generated yet.

Usage (from repo root):
    python scripts/epic_kitchen/download_audio.py
    python scripts/epic_kitchen/download_audio.py --data-dir ./data/epic_kitchen
    python scripts/epic_kitchen/download_audio.py --video-ids P01_01 P01_103
"""

import argparse
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.epic_kitchen.download_epic_kitchen.failure_log import FailureLog
from scripts.epic_kitchen.download_epic_kitchen.phase6_audio import download_audio


def _discover_video_ids(frames_root: Path) -> list[str]:
    """Return sorted list of video IDs inferred from the frames/ directory."""
    if not frames_root.exists():
        return []
    return sorted(p.name for p in frames_root.iterdir() if p.is_dir())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Re-run Phase 6: stream-extract audio for EPIC-Kitchens videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir",    default="./data/epic_kitchen",
                   help="Root data directory (frames/ and audio/ live here).")
    p.add_argument("--audio-dir",   default=None,
                   help="Override audio output dir (default: <data-dir>/audio).")
    p.add_argument("--failure-log", default=None,
                   help="Path to failure log (default: <data-dir>/failures_audio[_<job-index>].json).")
    p.add_argument("--video-ids",   nargs="+", default=None,
                   metavar="VIDEO_ID",
                   help="Explicit list of video IDs to process. "
                        "Defaults to all IDs found under frames/.")
    p.add_argument("--num-jobs",    type=int, default=1,
                   help="Total number of parallel SLURM array tasks.")
    p.add_argument("--job-index",   type=int, default=0,
                   help="0-based index of this task within the array (SLURM_ARRAY_TASK_ID).")
    return p.parse_args()


def main() -> None:
    args     = parse_args()
    data_dir = Path(args.data_dir)
    audio_root = Path(args.audio_dir) if args.audio_dir else data_dir / "audio"

    # Per-shard failure log when running as an array job.
    if args.failure_log:
        flog_path = Path(args.failure_log)
    elif args.num_jobs > 1:
        flog_path = data_dir / f"failures_audio_{args.job_index}.json"
    else:
        flog_path = data_dir / "failures_audio.json"

    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not found on PATH.", file=sys.stderr)
        sys.exit(1)

    if args.video_ids:
        video_ids = sorted(args.video_ids)
    else:
        video_ids = _discover_video_ids(data_dir / "frames")
        if not video_ids:
            print(f"No video directories found under {data_dir / 'frames'}. "
                  "Run phases 1-3 first, or pass --video-ids explicitly.",
                  file=sys.stderr)
            sys.exit(1)

    # Round-robin slice: job i handles indices i, i+num_jobs, i+2*num_jobs, …
    shard = video_ids[args.job_index::args.num_jobs]

    print("=" * 60)
    print("  Phase 6 — Audio extraction (standalone)")
    print(f"  job        : {args.job_index + 1}/{args.num_jobs}")
    print(f"  video IDs  : {len(shard)} of {len(video_ids)} total")
    print(f"  audio dir  : {audio_root}")
    print(f"  failure log: {flog_path}")
    print("=" * 60)

    sampled = [{"video_id": vid} for vid in shard]
    flog    = FailureLog(flog_path)
    download_audio(sampled, audio_root, flog)
    flog.flush()
    flog.summary()


if __name__ == "__main__":
    main()
