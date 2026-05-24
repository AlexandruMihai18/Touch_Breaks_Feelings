"""Launch run_all_evaluations.py for every registered prediction run.

Dataset-level shared config (annotations, clusters, dataset type) is defined
once in _DATASETS.  Each individual run only needs a csv path and a run_name.
Adding a new run is a single dict entry under the appropriate dataset key.

Usage
-----
    # Run everything
    python scripts/evaluation/run_batch_evaluations.py

    # Preview commands without executing
    python scripts/evaluation/run_batch_evaluations.py --dry-run

    # Run only specific run names
    python scripts/evaluation/run_batch_evaluations.py --runs ek_segGPT gh_onset

    # Run everything for one dataset
    python scripts/evaluation/run_batch_evaluations.py --datasets epic_kitchen

    # Override output root
    python scripts/evaluation/run_batch_evaluations.py --output-dir /scratch-shared/$USER/eval_results
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_EVAL_SCRIPT = Path(__file__).parent / "run_all_evaluations.py"

# ── Dataset-level shared config ───────────────────────────────────────────────

_DATASETS: dict[str, dict] = {
    "epic_kitchen": {
        "annotations": "/scratch-shared/dotero/epic_kitchen/annotations/val.json",
        "clusters":    "/home/dotero/Touch_Breaks_Feelings/data/epic_kitchen/object_clusters_7.json",
        "dataset":     "epic_kitchen",
    },
    "greatest_hits": {
        "annotations": "/scratch-shared/dotero/greatest_hits/annotations/val.json",
        "clusters":    "/home/dotero/Touch_Breaks_Feelings/data/epic_kitchen/object_clusters_gh.json",
        "dataset":     "greatest_hits",
    },
    "kubric": {
        "annotations": "/scratch-shared/dotero/kubric/annotations/val.json",
        "clusters":    None,
        "dataset":     "kubric",
    },
}

# ── Per-run registry ──────────────────────────────────────────────────────────
# Each entry: dataset key + csv path + run_name.
# To add a run, append a dict to the list.

_RUNS: list[dict] = [

    # ── Epic Kitchen — algorithmic ────────────────────────────────────────────
    dict(
        dataset  = "epic_kitchen",
        csv      = "/home/dotero/Touch_Breaks_Feelings/results/gabi/EPIC-SOUNDS_kmeans_audio_events_prediction_converted.csv",
        run_name = "ek_k_means",
    ),
    dict(
        dataset  = "epic_kitchen",
        csv      = "/home/dotero/Touch_Breaks_Feelings/results/gabi/EPIC-SOUNDS_onset_strength_threshold_prediction_converted.csv",
        run_name = "ek_onset",
    ),
    dict(
        dataset  = "epic_kitchen",
        csv      = "/home/dotero/Touch_Breaks_Feelings/results/gabi/EPIC-SOUNDS_rms_energy_threshold_prediction_converted.csv",
        run_name = "ek_rms_energy",
    ),

    # ── Epic Kitchen — SegGPT ─────────────────────────────────────────────────
    dict(
        dataset  = "epic_kitchen",
        csv      = "/home/dotero/Touch_Breaks_Feelings/results/epic_kitchen_val_seggpt_touch_results.csv",
        run_name = "ek_segGPT",
    ),

    # ── Epic Kitchen — Qwen ───────────────────────────────────────────────────
    dict(
        dataset  = "epic_kitchen",
        csv      = "/home/dotero/Touch_Breaks_Feelings/results/EpicKitchen_qwen2_audio_baseline_predictions.csv",
        run_name = "ek_qwen_audio",
    ),
    dict(
        dataset  = "epic_kitchen",
        csv      = "/home/dotero/Touch_Breaks_Feelings/results/EpicKitchen_qwen3_visual_audio_description_baseline_predictions.csv",
        run_name = "ek_qwen_vision_audio",
    ),
    dict(
        dataset  = "epic_kitchen",
        csv      = "/home/dotero/Touch_Breaks_Feelings/results/EpicKitchen_qwen3_visual_baseline_predictions.csv",
        run_name = "ek_qwen_vision",
    ),

    # ── Greatest Hits — algorithmic ───────────────────────────────────────────
    dict(
        dataset  = "greatest_hits",
        csv      = "/home/dotero/Touch_Breaks_Feelings/results/gabi/GreatestHits_kmeans_audio_events_prediction_converted.csv",
        run_name = "gh_k_means",
    ),
    dict(
        dataset  = "greatest_hits",
        csv      = "/home/dotero/Touch_Breaks_Feelings/results/gabi/GreatestHits_onset_strength_threshold_prediction_converted.csv",
        run_name = "gh_onset",
    ),
    dict(
        dataset  = "greatest_hits",
        csv      = "/home/dotero/Touch_Breaks_Feelings/results/gabi/GreatestHits_rms_energy_threshold_prediction_converted.csv",
        run_name = "gh_rms_energy",
    ),

    # ── Greatest Hits — SegGPT ────────────────────────────────────────────────
    dict(
        dataset  = "greatest_hits",
        csv      = "/home/dotero/Touch_Breaks_Feelings/results/greatest_hits_val_seggpt_touch_results.csv",
        run_name = "gh_segGPT",
    ),

    # ── Greatest Hits — Qwen ──────────────────────────────────────────────────
    dict(
        dataset  = "greatest_hits",
        csv      = "/home/dotero/Touch_Breaks_Feelings/results/GreatestHits_qwen3_visual_audio_description_baseline_predictions.csv",
        run_name = "gh_qwen_vision_audio",
    ),
    dict(
        dataset  = "greatest_hits",
        csv      = "/home/dotero/Touch_Breaks_Feelings/results/GreatestHits_qwen3_visual_baseline_predictions.csv",
        run_name = "gh_qwen_vision",
    ),
    dict(
        dataset  = "greatest_hits",
        csv      = "/home/dotero/Touch_Breaks_Feelings/results/GreatestHits_qwen2_audio_baseline_predictions.csv",
        run_name = "gh_qwen_audio",
    ),

    # ── Kubric ────────────────────────────────────────────────────────────────
    dict(
        dataset  = "kubric",
        csv      = "/home/dotero/Touch_Breaks_Feelings/results/Kubric_qwen3_visual_baseline_predictions.csv",
        run_name = "kubric_vision",
    ),
    dict(
        dataset  = "kubric",
        csv      = "/home/dotero/Touch_Breaks_Feelings/results/kubric_val_seggpt_touch_results.csv",
        run_name = "kubric_segGPT",
    ),

    # ── Epic Kitchen — DINO (merged binary + point) ───────────────────────────
    dict(
        dataset  = "epic_kitchen",
        csv      = "/home/dotero/Touch_Breaks_Feelings/results/ek_dino_mlp2_prediction.csv",
        run_name = "ek_dino_mlp2",
    ),
    dict(
        dataset  = "epic_kitchen",
        csv      = "/home/dotero/Touch_Breaks_Feelings/results/ek_dino_mlp3_prediction.csv",
        run_name = "ek_dino_mlp3",
    ),
    dict(
        dataset  = "epic_kitchen",
        csv      = "/home/dotero/Touch_Breaks_Feelings/results/ek_dino_mlp4_prediction.csv",
        run_name = "ek_dino_mlp4",
    ),
    dict(
        dataset  = "epic_kitchen",
        csv      = "/home/dotero/Touch_Breaks_Feelings/results/ek_dino_linear_prediction.csv",
        run_name = "ek_dino_linear",
    ),

    # ── Greatest Hits — DINO (merged binary + point) ─────────────────────────
    dict(
        dataset  = "greatest_hits",
        csv      = "/home/dotero/Touch_Breaks_Feelings/results/gh_dino_mlp2_prediction.csv",
        run_name = "gh_dino_mlp2",
    ),
    dict(
        dataset  = "greatest_hits",
        csv      = "/home/dotero/Touch_Breaks_Feelings/results/gh_dino_mlp3_prediction.csv",
        run_name = "gh_dino_mlp3",
    ),
    dict(
        dataset  = "greatest_hits",
        csv      = "/home/dotero/Touch_Breaks_Feelings/results/gh_dino_mlp4_prediction.csv",
        run_name = "gh_dino_mlp4",
    ),
    dict(
        dataset  = "greatest_hits",
        csv      = "/home/dotero/Touch_Breaks_Feelings/results/gh_dino_linear_prediction.csv",
        run_name = "gh_dino_linear",
    ),

    # ── Kubric — DINO (merged binary + point) ────────────────────────────────
    dict(
        dataset  = "kubric",
        csv      = "/home/dotero/Touch_Breaks_Feelings/results/kubric_dino_mlp2_prediction.csv",
        run_name = "kubric_dino_mlp2",
    ),
    dict(
        dataset  = "kubric",
        csv      = "/home/dotero/Touch_Breaks_Feelings/results/kubric_dino_mlp3_prediction.csv",
        run_name = "kubric_dino_mlp3",
    ),
    dict(
        dataset  = "kubric",
        csv      = "/home/dotero/Touch_Breaks_Feelings/results/kubric_dino_mlp4_prediction.csv",
        run_name = "kubric_dino_mlp4",
    ),
    dict(
        dataset  = "kubric",
        csv      = "/home/dotero/Touch_Breaks_Feelings/results/kubric_dino_linear_prediction.csv",
        run_name = "kubric_dino_linear",
    ),

    # ── Greatest Hits — Qwen finetuned (merged binary + point) ───────────────
    dict(
        dataset  = "greatest_hits",
        csv      = "/home/dotero/Touch_Breaks_Feelings/results/gh_qwen_ft_prediction.csv",
        run_name = "gh_qwen_ft",
    ),

    # ── Epic Kitchen — Qwen finetuned (binary only) ───────────────────────────
    dict(
        dataset  = "epic_kitchen",
        csv      = "/home/dotero/Touch_Breaks_Feelings/results/ek_qwen_ft_binary_prediction.csv",
        run_name = "ek_qwen_ft_binary",
    ),

    # ── Kubric — Qwen finetuned ────────────────────────────────────────────────
    dict(
        dataset  = "kubric",
        csv      = "/home/dotero/Touch_Breaks_Feelings/results/kubric_qwen_ft_binary_prediction.csv",
        run_name = "kubric_qwen_ft_binary",
    ),
    dict(
        dataset  = "kubric",
        csv      = "/home/dotero/Touch_Breaks_Feelings/results/kubric_qwen_ft_point_prediction.csv",
        run_name = "kubric_qwen_ft_point",
    ),
]


# ── Build subprocess command ──────────────────────────────────────────────────

def _build_cmd(run: dict, ds: dict, output_dir: Path, all_metrics: bool) -> list[str]:
    cmd = [
        sys.executable, str(_EVAL_SCRIPT),
        "--csv",         run["csv"],
        "--annotations", ds["annotations"],
        "--run-name",    run["run_name"],
        "--output-dir",  str(output_dir),
        "--dataset",     ds["dataset"],
    ]
    if ds["clusters"]:
        cmd += ["--clusters", ds["clusters"]]
    if all_metrics:
        cmd.append("--all-metrics")
    return cmd


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--runs", nargs="+", default=None, metavar="RUN_NAME",
        help="Run only these run names (default: all)",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        choices=list(_DATASETS), metavar="DATASET",
        help="Run only runs belonging to these datasets (default: all)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=_REPO / "results" / "evaluation",
        help="Parent directory for all run outputs (default: results/evaluation/)",
    )
    parser.add_argument(
        "--all-metrics", action="store_true",
        help="Pass --all-metrics to every evaluation run",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing them",
    )
    args = parser.parse_args()

    # Filter runs
    runs = _RUNS
    if args.datasets:
        runs = [r for r in runs if r["dataset"] in args.datasets]
    if args.runs:
        runs = [r for r in runs if r["run_name"] in args.runs]

    if not runs:
        print("No matching runs found.")
        return

    print(f"{'─' * 60}")
    print(f"  Batch evaluation  —  {len(runs)} run(s)")
    if args.dry_run:
        print("  [DRY RUN]")
    print(f"  Output root: {args.output_dir}")
    print(f"{'─' * 60}\n")

    passed = failed = 0

    for i, run in enumerate(runs, 1):
        ds = _DATASETS[run["dataset"]]
        cmd = _build_cmd(run, ds, args.output_dir, args.all_metrics)

        print(f"[{i}/{len(runs)}]  {run['run_name']}  ({run['dataset']})")

        csv_path = Path(run["csv"])
        if not csv_path.exists():
            print(f"  [SKIP] CSV not found: {csv_path}\n")
            failed += 1
            continue

        if args.dry_run:
            print("  " + " ".join(cmd) + "\n")
            continue

        result = subprocess.run(cmd)
        if result.returncode == 0:
            print(f"  [OK]\n")
            passed += 1
        else:
            print(f"  [FAILED]  exit code {result.returncode}\n")
            failed += 1

    if not args.dry_run:
        print(f"{'─' * 60}")
        print(f"  Done — passed={passed}  failed/skipped={failed}")
        print(f"  Outputs in: {args.output_dir}")
        print(f"{'─' * 60}")


if __name__ == "__main__":
    main()
