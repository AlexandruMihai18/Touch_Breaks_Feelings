"""Evaluate Qwen visual-audio and DINO-mlp3, then print a full RMSE comparison.

Phase 1 — Evaluate target runs:
  Runs only the point_zones step for each target run (fast — skips depth,
  object, coverage, zones, failures).  Qwen runs automatically receive
  --mask-mode zero_coord.  DINO runs use the default (predicted_touch).
  Pass --processor-size WxH to also compute RMSE in processor-output space
  for the Qwen runs (diagnostic).

Phase 2 — Collect & compare:
  Scans results/evaluation/*/point_heatmap_stats.json for every evaluated run
  (not just the ones just launched) and prints a single comparison table with
  RMSE, Hit@1, and N evaluated, grouped by dataset prefix.

Usage
-----
    python scripts/evaluation/run_rmse_comparison.py
    python scripts/evaluation/run_rmse_comparison.py --dry-run
    python scripts/evaluation/run_rmse_comparison.py --processor-size 448x448
    python scripts/evaluation/run_rmse_comparison.py --skip-eval
    python scripts/evaluation/run_rmse_comparison.py --output-dir /path/to/results
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO        = Path(__file__).resolve().parents[2]
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

# ── Target runs (Qwen visual-audio + DINO mlp3) ───────────────────────────────
# qwen=True  → --mask-mode zero_coord + optional --processor-size
# qwen=False → default mask mode (predicted_touch)

_TARGET_RUNS: list[dict] = [

    # ── Qwen visual-audio  (output in [0, 1000] scale) ────────────────────────
    dict(
        dataset     = "epic_kitchen",
        csv         = "/home/dotero/Touch_Breaks_Feelings/results/EpicKitchen_qwen3_visual_audio_description_baseline_predictions.csv",
        run_name    = "ek_qwen_vision_audio",
        mask_mode   = "zero_coord",
        coord_scale = 1000,
    ),
    dict(
        dataset     = "greatest_hits",
        csv         = "/home/dotero/Touch_Breaks_Feelings/results/GreatestHits_qwen3_visual_audio_description_baseline_predictions.csv",
        run_name    = "gh_qwen_vision_audio",
        mask_mode   = "zero_coord",
        coord_scale = 1000,
    ),

    # ── DINO mlp3  (output in 448×448 processor space) ───────────────────────
    dict(
        dataset     = "epic_kitchen",
        csv         = "/home/dotero/Touch_Breaks_Feelings/results/ek_dino_mlp3_prediction.csv",
        run_name    = "ek_dino_mlp3",
        mask_mode   = "predicted_touch",
        coord_scale = 448,
    ),
    dict(
        dataset     = "greatest_hits",
        csv         = "/home/dotero/Touch_Breaks_Feelings/results/gh_dino_mlp3_prediction.csv",
        run_name    = "gh_dino_mlp3",
        mask_mode   = "predicted_touch",
        coord_scale = 448,
    ),
    dict(
        dataset     = "kubric",
        csv         = "/home/dotero/Touch_Breaks_Feelings/results/kubric_dino_mlp3_prediction.csv",
        run_name    = "kubric_dino_mlp3",
        mask_mode   = "predicted_touch",
        coord_scale = 448,
    ),
]

# Steps irrelevant for RMSE — skipped to keep evaluation fast
_SKIP_STEPS = ["depth", "object", "coverage", "zones", "failures"]


# ── Subprocess helpers ────────────────────────────────────────────────────────

def _build_cmd(run: dict, ds: dict, output_dir: Path) -> list[str]:
    cmd = [
        sys.executable, str(_EVAL_SCRIPT),
        "--csv",         run["csv"],
        "--annotations", ds["annotations"],
        "--run-name",    run["run_name"],
        "--output-dir",  str(output_dir),
        "--dataset",     ds["dataset"],
        "--skip",        *_SKIP_STEPS,
        "--mask-mode",   run["mask_mode"],
        "--coord-scale", str(run["coord_scale"]),
    ]
    if ds["clusters"]:
        cmd += ["--clusters", ds["clusters"]]
    return cmd


def _run_one(run: dict, ds: dict, output_dir: Path) -> bool:
    csv_path = Path(run["csv"])
    if not csv_path.exists():
        print(f"  [SKIP] CSV not found: {csv_path}")
        return False

    cmd = _build_cmd(run, ds, output_dir)
    result = subprocess.run(cmd)
    ok = result.returncode == 0
    print(f"  [{'OK' if ok else 'FAILED'}]\n")
    return ok


# ── Table helpers ─────────────────────────────────────────────────────────────

# Canonical dataset prefix ordering for table grouping
_DS_ORDER = {"ek": 0, "gh": 1, "kubric": 2}


def _ds_prefix(run_name: str) -> str:
    return run_name.split("_")[0]


def _collect_stats(output_dir: Path) -> list[dict]:
    rows = []
    for stats_file in sorted(output_dir.glob("*/point_heatmap_stats.json")):
        run_name = stats_file.parent.name
        try:
            data = json.loads(stats_file.read_text())
        except Exception:
            continue
        n_total   = data.get("n", 0)
        n_masked  = data.get("n_masked", 0)
        rows.append({
            "run_name":            run_name,
            "n_eval":              n_total - n_masked,
            "n_total":             n_total,
            "rmse_img":            data.get("rmse_norm"),
            "rmse_proc":           data.get("rmse_norm_processor"),
            "processor_size":      data.get("processor_size"),
            "hit_at_1":            data.get("hit_at_1"),
            "mean_error":          data.get("mean_error_norm"),
            "mask_mode":           data.get("mask_mode", "?"),
            "gt_source":           data.get("gt_source", "?"),
        })
    return rows


def _fmt(v: float | None, decimals: int = 4) -> str:
    if v is None or (isinstance(v, float) and v != v):  # None or NaN
        return "—"
    return f"{v:.{decimals}f}"


def _print_table(rows: list[dict]) -> None:
    if not rows:
        print("  No point_heatmap_stats.json files found under output dir.")
        return

    rows.sort(key=lambda r: (_DS_ORDER.get(_ds_prefix(r["run_name"]), 99), r["run_name"]))

    has_proc = any(r["rmse_proc"] is not None for r in rows)

    name_w = max(len(r["run_name"]) for r in rows)
    name_w = max(name_w, 12)

    # Build header
    cols = [
        (f"{'Run':<{name_w}}", "l"),
        (f"{'N_eval':>8}",     "r"),
        (f"{'RMSE_img':>10}",  "r"),
    ]
    if has_proc:
        cols.append((f"{'RMSE_proc':>10}", "r"))
    cols += [
        (f"{'Hit@1':>7}", "r"),
        (f"{'MeanErr':>8}", "r"),
        ("  Mask", "l"),
    ]

    header = "  ".join(c for c, _ in cols)
    sep    = "─" * len(header)
    dbl    = "═" * len(header)

    print(f"\n{dbl}")
    print("  RMSE comparison — point touch prediction")
    print(dbl)
    print(header)
    print(sep)

    prev_prefix = None
    for r in rows:
        prefix = _ds_prefix(r["run_name"])
        if prev_prefix is not None and prefix != prev_prefix:
            print(sep)
        prev_prefix = prefix

        row_cols = [
            f"{r['run_name']:<{name_w}}",
            f"{r['n_eval']:>8,}",
            f"{_fmt(r['rmse_img']):>10}",
        ]
        if has_proc:
            row_cols.append(f"{_fmt(r['rmse_proc']):>10}")
        row_cols += [
            f"{_fmt(r['hit_at_1'], 3):>7}",
            f"{_fmt(r['mean_error']):>8}",
            f"  {r['mask_mode']}",
        ]
        print("  ".join(row_cols))

    print(dbl)
    if has_proc:
        proc_sizes = {r["processor_size"] for r in rows if r["processor_size"]}
        if proc_sizes:
            print(f"  RMSE_proc: processor-space normalization  ({', '.join(sorted(proc_sizes))})")
    print(f"  RMSE_img:  image-space normalization (original frame dims)")
    print(f"  N_eval:    label==1 samples after masking\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=_REPO / "results" / "evaluation",
        help="Evaluation output root (default: results/evaluation/)",
    )
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="Skip evaluation — just collect existing stats and print the table",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing them",
    )
    args = parser.parse_args()

    # ── Phase 1: evaluate target runs ─────────────────────────────────────────
    if not args.skip_eval:
        print(f"{'─' * 60}")
        print(f"  Phase 1 — evaluate {len(_TARGET_RUNS)} target run(s)")
        if args.dry_run:
            print("  [DRY RUN]")
        print(f"  Skipping steps: {', '.join(_SKIP_STEPS)}")
        print(f"{'─' * 60}\n")

        passed = failed = 0
        for i, run in enumerate(_TARGET_RUNS, 1):
            ds  = _DATASETS[run["dataset"]]
            print(f"[{i}/{len(_TARGET_RUNS)}]  {run['run_name']}  "
                  f"(mask={run['mask_mode']}  scale=÷{run['coord_scale']})")

            if args.dry_run:
                cmd = _build_cmd(run, ds, args.output_dir)
                print("  " + " ".join(cmd) + "\n")
                continue

            ok = _run_one(run, ds, args.output_dir)
            if ok:
                passed += 1
            else:
                failed += 1

        if not args.dry_run:
            print(f"{'─' * 60}")
            print(f"  Phase 1 done — passed={passed}  failed/skipped={failed}")
            print(f"{'─' * 60}\n")

    # ── Phase 2: collect all stats and print table ────────────────────────────
    print(f"  Phase 2 — scanning {args.output_dir} for point_heatmap_stats.json …")
    rows = _collect_stats(args.output_dir)
    print(f"  Found {len(rows)} run(s) with point predictions.\n")
    _print_table(rows)


if __name__ == "__main__":
    main()
