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
  * depth       — requires depth_touch field in the annotation JSONs
  * object      — requires --clusters and object_name field in the annotation JSONs
  * coverage    — requires object_coverage field in the annotation JSONs
  * zones       — requires x_touch / y_touch fields in the annotation JSONs
  * point_zones — auto-skipped when CSV has no predicted x_touch / y_touch coords

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
    "depth":       _EVAL / "depth"             / "analyze_depth_performance.py",
    "object":      _EVAL / "object_classes"    / "analyze_class_performance.py",
    "coverage":    _EVAL / "object_coverage"   / "analyze_coverage_performance.py",
    "zones":       _EVAL / "touch_zones"       / "analyze_grid_performance.py",
    "point_zones": _EVAL / "touch_zones"       / "analyze_point_performance.py",
    "failures":    _EVAL / "failure_sampling"  / "sample_failures.py",
}

_OUTPUTS = {
    "depth":       "depth_performance.png",
    "object":      "object_performance.png",
    "coverage":    "coverage_performance.png",
    "zones":       "grid_heatmap.png",
    "point_zones": "point_heatmap",
}


def _tee(msg: str, log: list[str]) -> None:
    print(msg)
    log.append(msg)


def _run(cmd: list[str], step: str, log: list[str]) -> bool:
    """Run a subprocess command, printing and logging stdout/stderr live."""
    _tee(f"\n{'─' * 60}", log)
    _tee(f"  [{step.upper()}]  {' '.join(str(c) for c in cmd)}", log)
    _tee(f"{'─' * 60}", log)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in proc.stdout:
        _tee(line.rstrip(), log)
    proc.wait()

    if proc.returncode != 0:
        _tee(f"  [WARN] {step} exited with code {proc.returncode} — continuing.", log)
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
                        help="Metric label for the depth bar chart — recall/accuracy only "
                             "because depth_touch exists only for touch-positive samples (default: recall)")
    parser.add_argument("--zones-metric", choices=["recall", "accuracy"], default="recall",
                        help="Metric label for the touch-zone heatmap — recall/accuracy only "
                             "because x/y_touch exists only for touch-positive samples (default: recall)")
    parser.add_argument("--coverage-metric",
                        choices=["recall", "precision", "f1", "accuracy"], default="f1",
                        help="Metric shown in the coverage bar chart (default: f1)")
    parser.add_argument("--object-metric",
                        choices=["recall", "precision", "f1", "accuracy"], default="f1",
                        help="Metric shown in the object-cluster bar chart (default: f1)")

    # Step selection
    parser.add_argument("--skip", nargs="*",
                        choices=list(_SCRIPTS), default=[],
                        metavar="STEP",
                        help="Steps to skip: depth object coverage zones point_zones failures")
    parser.add_argument("--all-metrics", action="store_true",
                        help="Pass --all-metrics to every analysis script: generates one figure per "
                             "metric with a _<metric> suffix. Per-script --*-metric flags are ignored.")

    # Point-touch normalization diagnostic
    parser.add_argument("--processor-size", type=str, default=None, metavar="WxH",
                        help="Passed to point_zones: if given (e.g. 448x448), also computes RMSE "
                             "in processor-output space and prints both for comparison.")

    # Failure sampling
    parser.add_argument("--dataset", choices=["epic_kitchen", "greatest_hits", "kubric"],
                        default=None,
                        help="Dataset type — enables mask overlay images in the failures step. "
                             "epic_kitchen: hand + object + refined-touch masks. "
                             "greatest_hits: stick + object + touch masks. "
                             "kubric: object1 + object2 + point-of-touch masks.")
    parser.add_argument("--n-samples", type=int, default=3,
                        help="Max failure samples per sub-category for the failures step (default: 3)")
    parser.add_argument("--failures-seed", type=int, default=42,
                        help="Random seed for failure sampling (default: 42)")

    args = parser.parse_args()

    if not args.csv.exists():
        parser.error(f"CSV not found: {args.csv}")

    # ── Create output directory ───────────────────────────────────────────────
    run_name = args.run_name or (
        args.csv.stem + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_dir: Path = args.output_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    log: list[str] = []

    for p in args.annotations:
        if not p.exists():
            _tee(f"[WARN] annotation file not found: {p}", log)

    _tee(f"\nOutputs → {out_dir}\n", log)

    py = sys.executable
    ann_args = [a for p in args.annotations for a in ("--annotations", str(p))]
    skipped = set(args.skip or [])
    all_m = args.all_metrics
    results: dict[str, str] = {}

    #  Save argparser output for reference
    with open(out_dir / "args.txt", "w") as f:
        f.write(" ".join(sys.argv) + "\n\n")
        f.write(textwrap.dedent(f"""\
            CSV: {args.csv}
            Annotations: {', '.join(str(p) for p in args.annotations)}
            Clusters: {args.clusters if args.clusters else 'None'}
            Output dir: {out_dir}
            Run name: {run_name}
            N bins: {args.n_bins}
            Grid size: {args.grid}
            Depth metric: {args.depth_metric}
            Zones metric: {args.zones_metric}
            Coverage metric: {args.coverage_metric}
            Object metric: {args.object_metric}
            Skip steps: {', '.join(skipped) if skipped else 'None'}
            All metrics flag: {all_m}
            Dataset: {args.dataset if args.dataset else 'None'}
            N failure samples: {args.n_samples}
            Failures seed: {args.failures_seed}
        """))

    # ── 1. Depth ──────────────────────────────────────────────────────────────
    if "depth" not in skipped:
        try:
            out = out_dir / _OUTPUTS["depth"]
            metric_args = ["--all-metrics"] if all_m else ["--metric", args.depth_metric]
            ok = _run(
                [py, str(_SCRIPTS["depth"]),
                 str(args.csv),
                 *metric_args,
                 "--output", str(out),
                 *ann_args],
                "depth", log,
            )
            results["depth"] = str(out_dir / "depth_performance_*.png") if (ok and all_m) else (str(out) if ok else "FAILED")
        except Exception as exc:
            _tee(f"  [ERROR] depth raised {exc!r}", log)
            results["depth"] = f"ERROR: {exc}"
    else:
        results["depth"] = "SKIPPED"

    # ── 2. Object clusters ────────────────────────────────────────────────────
    if "object" not in skipped:
        try:
            if args.clusters is None or not args.clusters.exists():
                _tee("\n[SKIP] object eval — --clusters not provided or file not found.", log)
                results["object"] = "SKIPPED (no --clusters)"
            else:
                out = out_dir / _OUTPUTS["object"]
                metric_args = ["--all-metrics"] if all_m else ["--metric", args.object_metric]
                ok = _run(
                    [py, str(_SCRIPTS["object"]),
                     str(args.csv),
                     "--clusters", str(args.clusters),
                     *metric_args,
                     "--output",   str(out),
                     *ann_args],
                    "object", log,
                )
                results["object"] = str(out_dir / "object_performance_*.png") if (ok and all_m) else (str(out) if ok else "FAILED")
        except Exception as exc:
            _tee(f"  [ERROR] object raised {exc!r}", log)
            results["object"] = f"ERROR: {exc}"
    else:
        results["object"] = "SKIPPED"

    # ── 3. Coverage ───────────────────────────────────────────────────────────
    if "coverage" not in skipped:
        try:
            out = out_dir / _OUTPUTS["coverage"]
            metric_args = ["--all-metrics"] if all_m else ["--metric", args.coverage_metric]
            ok = _run(
                [py, str(_SCRIPTS["coverage"]),
                 str(args.csv),
                 *metric_args,
                 "--output",  str(out),
                 *ann_args],
                "coverage", log,
            )
            results["coverage"] = str(out_dir / "coverage_performance_*.png") if (ok and all_m) else (str(out) if ok else "FAILED")
        except Exception as exc:
            _tee(f"  [ERROR] coverage raised {exc!r}", log)
            results["coverage"] = f"ERROR: {exc}"
    else:
        results["coverage"] = "SKIPPED"

    # ── 4. Touch zones ────────────────────────────────────────────────────────
    if "zones" not in skipped:
        try:
            out = out_dir / _OUTPUTS["zones"]
            metric_args = ["--all-metrics"] if all_m else ["--metric", args.zones_metric]
            ok = _run(
                [py, str(_SCRIPTS["zones"]),
                 str(args.csv),
                 "--grid",   str(args.grid),
                 *metric_args,
                 "--output", str(out),
                 *ann_args],
                "zones", log,
            )
            results["zones"] = str(out_dir / "grid_heatmap_*.png") if (ok and all_m) else (str(out) if ok else "FAILED")
        except Exception as exc:
            _tee(f"  [ERROR] zones raised {exc!r}", log)
            results["zones"] = f"ERROR: {exc}"
    else:
        results["zones"] = "SKIPPED"

    # ── 5. Touch-point regression zones ──────────────────────────────────────
    if "point_zones" not in skipped:
        try:
            import pandas as pd
            _x = pd.to_numeric(
                pd.read_csv(args.csv, usecols=["x_touch"])["x_touch"],
                errors="coerce",
            )
            if not _x.notna().any():
                _tee("\n[SKIP] point_zones — no predicted x_touch/y_touch coords in CSV", log)
                results["point_zones"] = "SKIPPED (no point predictions)"
            else:
                out = out_dir / _OUTPUTS["point_zones"]
                proc_args = ["--processor-size", args.processor_size] if args.processor_size else []
                ok = _run(
                    [py, str(_SCRIPTS["point_zones"]),
                     str(args.csv),
                     "--grid",   str(args.grid),
                     "--output", str(out),
                     *proc_args,
                     *ann_args],
                    "point_zones", log,
                )
                results["point_zones"] = str(out.with_name(out.name + "_hit_rate.png")) if ok else "FAILED"
        except Exception as exc:
            _tee(f"  [ERROR] point_zones raised {exc!r}", log)
            results["point_zones"] = f"ERROR: {exc}"
    else:
        results["point_zones"] = "SKIPPED"

    # ── 6. Failure sampling ───────────────────────────────────────────────────
    if "failures" not in skipped:
        try:
            dataset_args = ["--dataset", args.dataset] if args.dataset else []
            ok = _run(
                [py, str(_SCRIPTS["failures"]),
                 str(args.csv),
                 "--output-dir", str(out_dir),
                 "--n-samples",  str(args.n_samples),
                 "--seed",       str(args.failures_seed),
                 *dataset_args,
                 *ann_args],
                "failures", log,
            )
            results["failures"] = str(out_dir / "depth_failures") + ", " + str(out_dir / "object_size_failures") if ok else "FAILED"
        except Exception as exc:
            _tee(f"  [ERROR] failures raised {exc!r}", log)
            results["failures"] = f"ERROR: {exc}"
    else:
        results["failures"] = "SKIPPED"

    # ── Global metrics ────────────────────────────────────────────────────────
    try:
        import pandas as pd

        df = pd.read_csv(args.csv, usecols=["label", "prediction"])
        df["label"]      = pd.to_numeric(df["label"],      errors="coerce")
        df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
        df = df.dropna(subset=["label", "prediction"])

        tp = int(((df["label"] == 1) & (df["prediction"] == 1)).sum())
        tn = int(((df["label"] == 0) & (df["prediction"] == 0)).sum())
        fp = int(((df["label"] == 0) & (df["prediction"] == 1)).sum())
        fn = int(((df["label"] == 1) & (df["prediction"] == 0)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else float("nan"))
        accuracy  = (tp + tn) / len(df) if len(df) > 0 else float("nan")

        gm_lines = [
            f"Run: {run_name}",
            f"CSV: {args.csv}",
            "",
            "Confusion Matrix",
            f"  TP: {tp}",
            f"  TN: {tn}",
            f"  FP: {fp}",
            f"  FN: {fn}",
            f"  Total: {len(df)}",
            "",
            "Global Metrics",
            f"  Precision: {precision:.4f}",
            f"  Recall:    {recall:.4f}",
            f"  F1:        {f1:.4f}",
            f"  Accuracy:  {accuracy:.4f}",
        ]

        try:
            _iou = pd.to_numeric(
                pd.read_csv(args.csv, usecols=["iou"])["iou"],
                errors="coerce",
            ).dropna()
            if len(_iou) > 0:
                gm_lines.append(f"  Mean IoU:  {_iou.mean():.4f}")
        except (ValueError, KeyError):
            pass

        pt_stats_path = out_dir / "point_heatmap_stats.json"
        if pt_stats_path.exists():
            import json as _json
            pt = _json.loads(pt_stats_path.read_text())
            rmse = pt.get("rmse_norm", float("nan"))
            rmse_proc = pt.get("rmse_norm_processor")
            proc_size = pt.get("processor_size")
            n_masked = pt.get("n_masked", 0)
            n_total = pt.get("n", 0)
            mask_mode = pt.get("mask_mode", "predicted_touch")
            mode_label = "label=0 + zero-coord FN" if mask_mode == "zero_coord" else "label=0"
            gm_lines += [
                "",
                "Point Touch RMSE  (evaluated on label==1 samples)",
                f"  Mask mode: {mask_mode}  ({n_masked} {mode_label} excluded)",
                f"  RMSE image-normalized : {rmse:.4f}",
            ]
            if rmse_proc is not None:
                gm_lines.append(f"  RMSE processor-norm   : {rmse_proc:.4f}  "
                                 f"(processor size: {proc_size})")
            gm_lines.append(f"  N evaluated: {n_total - n_masked}  (of {n_total} total)")

        gm_path = out_dir / "global_metrics.txt"
        gm_path.write_text("\n".join(gm_lines) + "\n")
        _tee("\nGlobal Metrics", log)
        for line in gm_lines[3:]:
            _tee(line, log)
        _tee(f"\n  Global metrics → {gm_path}", log)
    except Exception as exc:
        _tee(f"  [WARN] could not compute global metrics: {exc!r}", log)

    # ── Summary ───────────────────────────────────────────────────────────────
    _tee(f"\n{'═' * 60}", log)
    _tee(f"  Evaluation complete — run: {run_name}", log)
    _tee(f"{'═' * 60}", log)
    for step, path in results.items():
        tag = "✓" if path not in ("FAILED", "SKIPPED") and not path.startswith("SKIPPED") else "–"
        _tee(f"  {tag}  {step:<12}  {path}", log)
    _tee(f"\n  All outputs in: {out_dir}\n", log)

    # ── Write log ─────────────────────────────────────────────────────────────
    log_path = out_dir / "evaluation_log.txt"
    log_path.write_text("\n".join(log) + "\n")
    print(f"  Log → {log_path}")


if __name__ == "__main__":
    main()
