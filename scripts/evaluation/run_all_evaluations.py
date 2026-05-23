"""Run all four evaluation analyses sequentially.

Outputs are written to results/evaluation/<run-name>/ (created automatically).

Usage
-----
    python scripts/evaluation/run_all_evaluations.py \\
        --csv results/predictions.csv \\
        --annotations /scratch-shared/$USER/epic_kitchen/annotations/val.json \\
        --clusters    data/epic_kitchen/object_clusters_7.json \\
        [--run-name   my_experiment] \\
        [--output-dir results/evaluation] \\
        [--n-bins 5] [--grid 8] [--metric f1]

Skipped analyses
----------------
  * depth    — requires depth_touch field in the annotation JSONs
  * object   — requires --clusters and object_name field in the annotation JSONs
  * coverage — requires object_coverage field in the annotation JSONs
  * zones    — requires x_touch / y_touch fields in the annotation JSONs

Each step prints "[SKIP] ..." if the required data is absent; other steps
continue normally.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path

# ── Repo root (two levels up from this file) ─────────────────────────────────
_REPO = Path(__file__).resolve().parents[2]
_EVAL = Path(__file__).parent

_SCRIPTS = {
    "depth":    _EVAL / "depth"           / "analyze_depth_performance.py",
    "object":   _EVAL / "object_classes"  / "analyze_object_performance.py",
    "coverage": _EVAL / "object_coverage" / "analyze_coverage_performance.py",
    "zones":    _EVAL / "touch_zones"     / "analyze_grid_performance.py",
}

_OUTPUTS = {
    "depth":    "depth_performance.png",
    "object":   "object_performance.png",
    "coverage": "coverage_performance.png",
    "zones":    "grid_heatmap.png",
}


def _run(cmd: list[str], step: str) -> bool:
    """Run a subprocess command, print stdout/stderr live.  Returns True on success."""
    print(f"\n{'─' * 60}")
    print(f"  [{step.upper()}]  {' '.join(str(c) for c in cmd)}")
    print(f"{'─' * 60}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"  [WARN] {step} exited with code {result.returncode} — continuing.")
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required
    parser.add_argument("--csv", type=Path, required=True,
                        help="Predictions CSV (must contain label, prediction, frame_path)")
    parser.add_argument("--annotations", nargs="+", type=Path, required=True,
                        help="Annotation JSON file(s) — used by all four analyses")

    # Output
    parser.add_argument("--run-name", default=None,
                        help="Subdirectory name inside --output-dir (default: CSV stem + timestamp)")
    parser.add_argument("--output-dir", type=Path,
                        default=_REPO / "results" / "evaluation",
                        help="Parent directory for run output (default: results/evaluation/)")

    # Object-cluster eval
    parser.add_argument("--clusters", type=Path, default=None,
                        help="Cluster mapping JSON required by the object analysis")

    # Per-analysis tuning
    parser.add_argument("--n-bins", type=int, default=5,
                        help="Quantile bins for depth and coverage analyses (default: 5)")
    parser.add_argument("--grid", type=int, default=8,
                        help="Grid dimension for the touch-zone heatmap (default: 8)")
    parser.add_argument("--depth-metric", choices=["recall", "accuracy"], default="recall",
                        help="Metric label for the depth bar chart (default: recall)")
    parser.add_argument("--zones-metric", choices=["recall", "accuracy"], default="recall",
                        help="Metric label for the touch-zone heatmap (default: recall)")
    parser.add_argument("--coverage-metric", choices=["f1", "accuracy"], default="f1",
                        help="Metric shown in the coverage bar chart (default: f1)")

    # Step selection
    parser.add_argument("--skip", nargs="*",
                        choices=list(_SCRIPTS), default=[],
                        metavar="STEP",
                        help="Steps to skip: depth object coverage zones")

    args = parser.parse_args()

    if not args.csv.exists():
        parser.error(f"CSV not found: {args.csv}")
    for p in args.annotations:
        if not p.exists():
            print(f"[WARN] annotation file not found: {p}")

    # ── Create output directory ───────────────────────────────────────────────
    run_name = args.run_name or (
        args.csv.stem + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_dir: Path = args.output_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutputs → {out_dir}\n")

    py = sys.executable
    ann_args = [a for p in args.annotations for a in ("--annotations", str(p))]
    skipped = set(args.skip or [])
    results: dict[str, str] = {}

    # ── 1. Depth ──────────────────────────────────────────────────────────────
    if "depth" not in skipped:
        out = out_dir / _OUTPUTS["depth"]
        ok = _run(
            [py, str(_SCRIPTS["depth"]),
             str(args.csv),
             "--n-bins", str(args.n_bins),
             "--metric", args.depth_metric,
             "--output", str(out),
             *ann_args],
            "depth",
        )
        results["depth"] = str(out) if ok else "FAILED"
    else:
        results["depth"] = "SKIPPED"

    # ── 2. Object clusters ────────────────────────────────────────────────────
    if "object" not in skipped:
        if args.clusters is None or not args.clusters.exists():
            print("\n[SKIP] object eval — --clusters not provided or file not found.")
            results["object"] = "SKIPPED (no --clusters)"
        else:
            out = out_dir / _OUTPUTS["object"]
            ok = _run(
                [py, str(_SCRIPTS["object"]),
                 str(args.csv),
                 "--clusters", str(args.clusters),
                 "--output",   str(out),
                 *ann_args],
                "object",
            )
            results["object"] = str(out) if ok else "FAILED"
    else:
        results["object"] = "SKIPPED"

    # ── 3. Coverage ───────────────────────────────────────────────────────────
    if "coverage" not in skipped:
        out = out_dir / _OUTPUTS["coverage"]
        ok = _run(
            [py, str(_SCRIPTS["coverage"]),
             str(args.csv),
             "--n-bins", str(args.n_bins),
             "--metric",  args.metric,
             "--output",  str(out),
             *ann_args],
            "coverage",
        )
        results["coverage"] = str(out) if ok else "FAILED"
    else:
        results["coverage"] = "SKIPPED"

    # ── 4. Touch zones ────────────────────────────────────────────────────────
    if "zones" not in skipped:
        out = out_dir / _OUTPUTS["zones"]
        ok = _run(
            [py, str(_SCRIPTS["zones"]),
             str(args.csv),
             "--grid",   str(args.grid),
             "--metric", args.zones_metric,
             "--output", str(out),
             *ann_args],
            "zones",
        )
        results["zones"] = str(out) if ok else "FAILED"
    else:
        results["zones"] = "SKIPPED"

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  Evaluation complete — run: {run_name}")
    print(f"{'═' * 60}")
    for step, path in results.items():
        tag = "✓" if path not in ("FAILED", "SKIPPED") and not path.startswith("SKIPPED") else "–"
        print(f"  {tag}  {step:<12}  {path}")
    print(f"\n  All outputs in: {out_dir}\n")


if __name__ == "__main__":
    main()
