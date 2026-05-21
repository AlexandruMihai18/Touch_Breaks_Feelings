"""
Run the Greatest Hits data pipeline end-to-end.

Steps and their dependencies:

  download          1_download_greatest_hits.sh              (no deps)
  extract           2_extract_greatest_hits_frames.py        (needs: download)
  context-frames    2_1_generate_context_frames.py           (needs: extract)
                    → writes train/val_context_frames.json
  gt-annotations    3_generate_gh_gt_annotations.py          (needs: extract)
                    → writes train/val annotation JSONs
  annotate          4_annotate_greatest_hits.py              (needs: extract, GPU)
  mask-annotations  5_generate_gh_mask_annotations.py        (needs: gt-annotations + annotate)
  evaluate          6_evaluate_annotation_pipeline.py        (needs: gt-annotations + annotate)

Usage
-----
    # Full pipeline (download → evaluate)
    python scripts/greatest_hits/run_pipeline.py --download-mode low

    # Skip download, run everything else
    python scripts/greatest_hits/run_pipeline.py --skip download

    # Run only the annotation-generation steps
    python scripts/greatest_hits/run_pipeline.py --steps gt-annotations mask-annotations

    # Dry run: print commands without executing
    python scripts/greatest_hits/run_pipeline.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).parent
_PYTHON = sys.executable

ALL_STEPS = [
    "download",
    "extract",
    "context-frames",
    "gt-annotations",
    "annotate",
    "mask-annotations",
    "evaluate",
]

_GPU_STEPS = {"annotate"}
_ANALYSIS_STEPS = {"evaluate"}


def _cmd(step: str, args: argparse.Namespace) -> list[str]:
    """Return the subprocess command for the given step."""
    if step == "download":
        return [
            "bash",
            str(_HERE / "1_download_greatest_hits.sh"),
            f"--{args.download_mode}",
        ]
    if step == "extract":
        return [
            _PYTHON,
            str(_HERE / "2_extract_greatest_hits_frames.py"),
            "--data_dir",   args.data_dir,
            "--output_dir", args.frames_dir,
            "--num_workers", str(args.workers),
            "--skip_processed_videos",
        ]
    if step == "context-frames":
        return [
            _PYTHON,
            str(_HERE / "2_1_generate_context_frames.py"),
            "--data-dir",   args.data_dir,
            "--frames-dir", args.frames_dir,
            "--output-dir", args.annotations_dir,
        ]
    if step == "gt-annotations":
        return [
            _PYTHON,
            str(_HERE / "3_generate_gh_gt_annotations.py"),
            "--data-dir",   args.data_dir,
            "--frames-dir", args.frames_dir,
            "--output-dir", args.annotations_dir,
        ]
    if step == "annotate":
        return [
            _PYTHON,
            str(_HERE / "4_annotate_greatest_hits.py"),
            "--skip-existing",
        ]
    if step == "mask-annotations":
        return [
            _PYTHON,
            str(_HERE / "5_generate_gh_mask_annotations.py"),
            "--masks-dir",       args.masks_dir,
            "--annotations-dir", args.annotations_dir,
        ]
    if step == "evaluate":
        return [
            _PYTHON,
            str(_HERE / "6_evaluate_annotation_pipeline.py"),
            "--masks-dir",       args.masks_dir,
            "--annotations-dir", args.annotations_dir,
            "--output",          str(Path(args.output_dir) / "gh_eval.csv"),
        ]
    raise ValueError(f"Unknown step: {step}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the Greatest Hits pipeline end-to-end.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--steps", nargs="+", default=ALL_STEPS, metavar="STEP",
        choices=ALL_STEPS,
        help="Steps to run (default: all). Choices: " + ", ".join(ALL_STEPS),
    )
    p.add_argument(
        "--skip", nargs="+", default=[], metavar="STEP",
        choices=ALL_STEPS,
        help="Steps to skip even if listed in --steps.",
    )
    p.add_argument(
        "--download-mode", default="low", choices=["low", "full", "features", "all"],
        help="Resolution to download (only used if 'download' step is active).",
    )
    p.add_argument(
        "--data-dir",        default="data/greatest_hits/vis-data-256/vis-data-256",
        help="Directory containing *_denoised.mp4 and *_times.txt files.",
    )
    p.add_argument(
        "--frames-dir",      default="data/greatest_hits/frames",
        help="Root directory for extracted frames.",
    )
    p.add_argument(
        "--masks-dir",       default="data/greatest_hits/masks",
        help="Root directory for annotation masks.",
    )
    p.add_argument(
        "--annotations-dir", default="data/greatest_hits/annotations",
        help="Directory to write train.json / val.json.",
    )
    p.add_argument(
        "--output-dir",      default="results",
        help="Directory for evaluation CSVs.",
    )
    p.add_argument(
        "--workers", type=int, default=8,
        help="Parallel workers for the extract step.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing them.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    skip = set(args.skip)
    steps = [s for s in args.steps if s not in skip]

    gpu_steps = [s for s in steps if s in _GPU_STEPS]
    if gpu_steps and not args.dry_run:
        print(f"  Note: {', '.join(gpu_steps)} require a GPU.")

    print("=" * 60)
    print("  Greatest Hits pipeline")
    print(f"  Steps : {' → '.join(steps)}")
    print(f"  Mode  : {'DRY RUN' if args.dry_run else 'EXECUTE'}")
    print("=" * 60)

    for step in steps:
        cmd = _cmd(step, args)
        label = f"[{step}]"
        print(f"\n{'─' * 60}")
        print(f"  {label}  {' '.join(str(c) for c in cmd)}")
        print(f"{'─' * 60}")

        if args.dry_run:
            continue

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\nStep '{step}' failed (exit {result.returncode}). Stopping.")
            sys.exit(result.returncode)

    if args.dry_run:
        print("\n[dry-run] No commands were executed.")
    else:
        print("\nAll steps completed successfully.")


if __name__ == "__main__":
    main()
