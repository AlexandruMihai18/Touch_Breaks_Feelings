"""
Download binary performance metric files from a wandb run.

The BinaryMetricsCallback saves two JSON files per run:
  binary_performance_metrics/best.json   — metrics at the best val epoch
  binary_performance_metrics/final.json  — metrics at the last val epoch

This script downloads both (or just the one specified via --variant) to a
local directory, printing the global metrics summary on download.

Usage:
    python scripts/download_binary_metrics.py <run_id>
    python scripts/download_binary_metrics.py <run_id> --variant best
    python scripts/download_binary_metrics.py <run_id> --output results/metrics/
    python scripts/download_binary_metrics.py <run_id> --entity myteam --project myproject
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import wandb


_DEFAULT_ENTITY  = "touchgpt"
_DEFAULT_PROJECT = "touchgpt"
_REMOTE_FOLDER   = "binary_performance_metrics"
_VARIANTS        = ("best", "final")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _download_file(run, remote_name: str, dest: Path) -> bool:
    """Download one file from a wandb run. Returns False if not found."""
    try:
        run.file(remote_name).download(root=str(dest.parent), replace=True)
        downloaded = dest.parent / remote_name
        downloaded.rename(dest)
        return True
    except wandb.errors.CommError as e:
        if "404" in str(e):
            return False
        raise


def download_metrics(
    run_id: str,
    variants: list[str],
    output_dir: Path,
    entity: str = _DEFAULT_ENTITY,
    project: str = _DEFAULT_PROJECT,
) -> dict[str, Path]:
    """Download metric JSON files from a wandb run.

    Returns a mapping of variant → local path for successfully downloaded files.
    """
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    print(f"Run: {run.name}  ({entity}/{project}/{run_id})")
    print(f"State: {run.state}  |  Epochs: {run.summary.get('epoch', '?')}")

    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Path] = {}

    with tempfile.TemporaryDirectory() as tmp:
        for variant in variants:
            remote = f"{_REMOTE_FOLDER}/{variant}.json"
            dest   = output_dir / f"{variant}.json"
            tmp_dest = Path(tmp) / f"{variant}.json"

            print(f"\n  Downloading {remote} …")
            if _download_file(run, remote, tmp_dest):
                tmp_dest.rename(dest)
                results[variant] = dest
                print(f"  → {dest}")
            else:
                print(f"  ✗ Not found (run may not have completed a validation epoch)")

    return results


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def _fmt(v: float | None, pct: bool = True) -> str:
    if v is None:
        return "  N/A"
    return f"{v * 100:5.1f}%" if pct else f"{v:.4f}"


def print_summary(path: Path) -> None:
    data = json.loads(path.read_text())
    print(f"\n{'─' * 60}")
    print(f"  {path.name}  (epoch={data.get('epoch', '?')}  step={data.get('step', '?')})")
    print(f"{'─' * 60}")
    for split in ("train", "val"):
        g = data.get(split, {}).get("global", {})
        print(
            f"  {split:<6}  n={g.get('n', 0):>7,}  "
            f"TP={g.get('tp', 0):>6,}  FP={g.get('fp', 0):>6,}  "
            f"FN={g.get('fn', 0):>6,}  TN={g.get('tn', 0):>6,}  "
            f"Acc={_fmt(g.get('accuracy'))}  "
            f"P={_fmt(g.get('precision'))}  "
            f"R={_fmt(g.get('recall'))}  "
            f"F1={_fmt(g.get('f1'))}"
        )

    # Top-10 val classes by sample count
    per_class = data.get("val", {}).get("per_class", {})
    if per_class:
        top = sorted(per_class.items(), key=lambda kv: -(kv[1].get("n") or 0))[:10]
        print(f"\n  Top-10 val classes by n:")
        for cls, m in top:
            print(
                f"    {cls:<35}  n={m.get('n', 0):>5,}  "
                f"F1={_fmt(m.get('f1'))}  R={_fmt(m.get('recall'))}"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download binary performance metrics from a wandb run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("run_id", help="wandb run ID (e.g. abc123xy).")
    p.add_argument(
        "--variant", choices=list(_VARIANTS) + ["all"], default="all",
        help="Which file(s) to download.",
    )
    p.add_argument(
        "--output", type=Path, default=Path("results/binary_metrics"),
        help="Local directory to write the downloaded JSON files.",
    )
    p.add_argument("--entity",  default=_DEFAULT_ENTITY,  help="wandb entity.")
    p.add_argument("--project", default=_DEFAULT_PROJECT, help="wandb project.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    variants = list(_VARIANTS) if args.variant == "all" else [args.variant]
    out_dir  = args.output / args.run_id

    downloaded = download_metrics(
        run_id=args.run_id,
        variants=variants,
        output_dir=out_dir,
        entity=args.entity,
        project=args.project,
    )

    if not downloaded:
        print("\nNo files downloaded.")
        sys.exit(1)

    for variant, path in downloaded.items():
        print_summary(path)

    print(f"\nFiles saved to: {out_dir}")


if __name__ == "__main__":
    main()
