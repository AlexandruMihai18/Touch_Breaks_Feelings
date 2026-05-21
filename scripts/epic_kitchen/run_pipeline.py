"""
Run the EPIC Kitchen data pipeline end-to-end.

Steps:

  download    download_epic_kitchen/         — download VISOR, extract frames,
                                               render masks, generate annotations
  refine      refine_touch_masks.py          — depth-filtered touch masks (GPU)
  evaluate    evaluate_refined_masks.py      — binary evaluation report (optional)

Usage
-----
    # Full pipeline
    python scripts/epic_kitchen/run_pipeline.py

    # Skip download (data already on disk)
    python scripts/epic_kitchen/run_pipeline.py --skip download

    # Only the annotation + refinement steps
    python scripts/epic_kitchen/run_pipeline.py --steps download refine

    # Dry run: print commands without executing
    python scripts/epic_kitchen/run_pipeline.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).parent
_PYTHON = sys.executable

ALL_STEPS = ["download", "refine", "evaluate"]
_GPU_STEPS = {"refine"}


def _cmd(step: str, args: argparse.Namespace) -> list[str]:
    if step == "download":
        return [
            _PYTHON,
            str(_HERE / "download_epic_kitchen"),
            "--all-participants",
            "--match-no-contact",
            "--split",      "train",
            "--workers",    str(args.workers),
            "--output-dir", args.data_dir,
        ]
    if step == "refine":
        return [
            _PYTHON,
            str(_HERE / "refine_touch_masks.py"),
            "--splits", "train", "val",
        ]
    if step == "evaluate":
        return [
            _PYTHON,
            str(_HERE / "evaluate_refined_masks.py"),
            "--splits", "train", "val",
            "--output", str(Path(args.output_dir) / "ek_refined_eval.csv"),
        ]
    raise ValueError(f"Unknown step: {step}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the EPIC Kitchen pipeline end-to-end.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--steps", nargs="+", default=ALL_STEPS, metavar="STEP",
        choices=ALL_STEPS,
        help=f"Steps to run. Choices: {', '.join(ALL_STEPS)}",
    )
    p.add_argument(
        "--skip", nargs="+", default=[], metavar="STEP",
        choices=ALL_STEPS,
        help="Steps to skip even if listed in --steps.",
    )
    p.add_argument(
        "--data-dir", default="data/epic_kitchen",
        help="Root data directory passed to download_epic_kitchen.",
    )
    p.add_argument(
        "--output-dir", default="results",
        help="Directory for evaluation CSVs.",
    )
    p.add_argument(
        "--workers", type=int, default=8,
        help="Parallel workers for the download step.",
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
    print("  EPIC Kitchen pipeline")
    print(f"  Steps : {' → '.join(steps)}")
    print(f"  Mode  : {'DRY RUN' if args.dry_run else 'EXECUTE'}")
    print("=" * 60)

    for step in steps:
        cmd = _cmd(step, args)
        print(f"\n{'─' * 60}")
        print(f"  [{step}]  {' '.join(str(c) for c in cmd)}")
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
