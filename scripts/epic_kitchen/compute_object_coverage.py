#!/usr/bin/env python3
"""
compute_object_coverage.py — rank object categories by median mask coverage.

For each sample in the EPIC Kitchen annotations, opens the object mask and
computes coverage = (nonzero pixels) / (total pixels).  Results are grouped
by object_name and summarised with median, mean, and sample count.

Outputs:
    data/epic_kitchen/object_labels_coverage.txt         ← all categories, sorted by median
    data/epic_kitchen/object_labels_filtered.txt         ← non-surface categories above --min-coverage
    data/epic_kitchen/object_labels_surface_removed.txt  ← surface/structural categories (hard-excluded)
    data/epic_kitchen/object_labels_removed.txt          ← non-surface categories below --min-coverage

Usage:
    python scripts/epic_kitchen/compute_object_coverage.py
    python scripts/epic_kitchen/compute_object_coverage.py --min-coverage 0.01 --workers 8
    python scripts/epic_kitchen/compute_object_coverage.py --splits train val
"""

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

# Categories that represent architectural surfaces, structural frames, or
# degenerate background labels. Their masks span large flat regions with no
# clear depth boundary from the hand, making touch annotations unreliable
# regardless of coverage value. They are written to a separate output file
# instead of being mixed with the coverage-threshold rejects.
SURFACE_LABELS: frozenset[str] = frozenset({
    # floors / walls / windows
    "floor",
    "kitchen floor",
    "wall",
    "window",
    # flat working surfaces
    "surface",
    "counter",
    "table",
    "desk",
    "stove top",
    # shelves and rack surfaces
    "shelf",
    "oven shelf",
    "steaming shelf",
    # surface coverings
    "table cloth",
    "place mat",
    # structural frames / large background objects
    "ladder",
    "clothes horse",
    "grid",
    # generic flat labels (specific variants like "chopping board" are kept)
    "board",
    "top",
    "base",
    # the kitchen room itself
    "kitchen",
})


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute per-category object mask coverage statistics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir",     default="./data/epic_kitchen")
    p.add_argument("--splits",       nargs="+", default=["train", "val"])
    p.add_argument("--min-coverage", type=float, default=0.005,
                   help="Minimum median pixel coverage to include in filtered list (0–1)")
    p.add_argument("--workers",      type=int,   default=8)
    return p.parse_args()


_pair_meta_cache: dict[Path, dict] = {}


def _read_pair_meta(mask_dir: Path) -> dict:
    if mask_dir not in _pair_meta_cache:
        p = mask_dir / "pair_metadata.json"
        _pair_meta_cache[mask_dir] = json.loads(p.read_text()) if p.exists() else {}
    return _pair_meta_cache[mask_dir]


def _object_name(sample: dict) -> str:
    """Return object_name from the annotation entry or pair_metadata.json fallback."""
    raw  = sample.get("object_name") or ""
    name = str(raw).strip() if raw else ""
    if name:
        return name
    mask_path = Path(sample.get("object_mask_path", ""))
    if not mask_path.name:
        return ""
    # stem: "{frame}_p{i}_object"  →  key: "{frame}_p{i}"
    key = mask_path.stem.removesuffix("_object")
    meta = _read_pair_meta(mask_path.parent)
    return meta.get(key, {}).get("object_name", "").strip()


def _coverage(mask_path: str) -> float | None:
    try:
        arr = np.array(Image.open(mask_path).convert("L"))
        return float(np.count_nonzero(arr)) / arr.size
    except Exception:
        return None


def main():
    args    = parse_args()
    data    = Path(args.data_dir)
    anno    = data / "annotations"

    # ── load all samples ────────────────────────────────────────────────────
    samples = []
    for split in args.splits:
        f = anno / f"{split}.json"
        if not f.exists():
            print(f"  ⚠ {f} not found — skipping")
            continue
        loaded = json.loads(f.read_text())
        samples.extend(loaded)
        print(f"  {split}: {len(loaded):,} samples")

    if not samples:
        print("No samples found.")
        return

    with_object = [s for s in samples if s.get("object_mask_path")]
    print(f"\n  {len(with_object):,} samples with object mask")

    # ── compute coverage in parallel ────────────────────────────────────────
    print(f"\nComputing object mask coverage ({args.workers} workers)…")
    coverages: dict[str, list[float]] = defaultdict(list)
    total = len(with_object)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(_coverage, s["object_mask_path"]): s
            for s in with_object
        }
        for i, fut in enumerate(as_completed(futs), 1):
            sample = futs[fut]
            name   = _object_name(sample) or "_unknown"
            cov    = fut.result()
            if cov is not None:
                coverages[name].append(cov)
            if i % 500 == 0 or i == total:
                print(f"  {i}/{total}", end="\r")

    unknown = len(coverages.pop("_unknown", []))
    if unknown:
        print(f"\n  ⚠ {unknown} samples had no object_name (excluded from per-category stats)")

    print()

    # ── aggregate ───────────────────────────────────────────────────────────
    stats = []
    for name, covs in coverages.items():
        arr = np.array(covs)
        stats.append({
            "name":   name,
            "count":  len(arr),
            "median": float(np.median(arr)),
            "mean":   float(np.mean(arr)),
            "p10":    float(np.percentile(arr, 10)),
            "p90":    float(np.percentile(arr, 90)),
        })

    stats.sort(key=lambda x: -x["median"])

    # ── write coverage report ────────────────────────────────────────────────
    cov_out = data / "object_labels_coverage.txt"
    header  = (
        f"# EPIC-KITCHENS object mask coverage\n"
        f"# {len(stats)} categories · splits: {', '.join(args.splits)}\n"
        f"# coverage = nonzero pixels / total pixels\n"
        f"#\n"
        f"# {'median':>7}  {'mean':>7}  {'p10':>7}  {'p90':>7}  {'n':>5}  name\n"
    )
    lines = [
        f"  {s['median']:7.4f}  {s['mean']:7.4f}  {s['p10']:7.4f}  {s['p90']:7.4f}  {s['count']:>5}  {s['name']}"
        for s in stats
    ]
    cov_out.write_text(header + "\n".join(lines) + "\n")
    print(f"\nCoverage report → {cov_out}")

    # ── split surface labels out before applying the coverage threshold ─────
    surface_stats  = [s for s in stats if s["name"] in SURFACE_LABELS]
    non_surface    = [s for s in stats if s["name"] not in SURFACE_LABELS]

    kept    = [s for s in non_surface if s["median"] >= args.min_coverage]
    removed = [s for s in non_surface if s["median"] <  args.min_coverage]

    # ── write filtered list (above threshold, surfaces excluded) ─────────────
    filt_out = data / "object_labels_filtered.txt"
    filt_header = (
        f"# EPIC-KITCHENS filtered object labels\n"
        f"# {len(kept)} categories with median mask coverage >= {args.min_coverage:.4f}\n"
        f"# (removed {len(removed)} below threshold, {len(surface_stats)} surface labels in separate file)\n"
        f"# sorted by median coverage descending\n"
        f"#\n"
        f"# count  name\n"
    )
    filt_lines = [f"{s['count']:>6}  {s['name']}" for s in kept]
    filt_out.write_text(filt_header + "\n".join(filt_lines) + "\n")
    print(f"Filtered list       → {filt_out}  ({len(kept)} kept, {len(removed)} removed)")

    # ── write surface-removed list ────────────────────────────────────────────
    surf_out = data / "object_labels_surface_removed.txt"
    surf_header = (
        f"# EPIC-KITCHENS surface-removed object labels\n"
        f"# {len(surface_stats)} categories excluded as architectural surfaces or structural frames\n"
        f"# These are removed regardless of coverage because their masks span large flat\n"
        f"# regions with no reliable depth boundary from the hand.\n"
        f"# sorted by median coverage descending\n"
        f"#\n"
        f"# {'median':>7}  {'mean':>7}  {'n':>5}  name\n"
    )
    surf_lines = [
        f"  {s['median']:7.4f}  {s['mean']:7.4f}  {s['count']:>5}  {s['name']}"
        for s in sorted(surface_stats, key=lambda x: -x["median"])
    ]
    surf_out.write_text(surf_header + "\n".join(surf_lines) + "\n")
    print(f"Surface-removed     → {surf_out}  ({len(surface_stats)} categories)")

    # ── write coverage-threshold removed list ─────────────────────────────────
    rm_out = data / "object_labels_removed.txt"
    rm_header = (
        f"# EPIC-KITCHENS removed object labels (coverage threshold)\n"
        f"# {len(removed)} non-surface categories with median mask coverage < {args.min_coverage:.4f}\n"
        f"# sorted by median coverage descending\n"
        f"#\n"
        f"# {'median':>7}  {'n':>5}  name\n"
    )
    rm_lines = [f"  {s['median']:7.4f}  {s['count']:>5}  {s['name']}" for s in removed]
    rm_out.write_text(rm_header + "\n".join(rm_lines) + "\n")
    print(f"Coverage-removed    → {rm_out}")

    # ── summary ──────────────────────────────────────────────────────────────
    all_medians = [s["median"] for s in stats]
    if not all_medians:
        print("\nNo categories with coverage data found. "
              "Check that object_mask_path fields point to existing files.")
        return
    print(f"\nCoverage across all categories:")
    print(f"  p10  = {np.percentile(all_medians, 10):.4f}")
    print(f"  p25  = {np.percentile(all_medians, 25):.4f}")
    print(f"  p50  = {np.percentile(all_medians, 50):.4f}")
    print(f"  p75  = {np.percentile(all_medians, 75):.4f}")
    print(f"  p90  = {np.percentile(all_medians, 90):.4f}")
    ns_medians = [s["median"] for s in non_surface]
    if ns_medians:
        below_pct = 100 * sum(m < args.min_coverage for m in ns_medians) / len(ns_medians)
        print(f"\nThreshold {args.min_coverage:.4f} removes {below_pct:.0f}% of non-surface categories.")
    print(f"Surface labels hard-excluded: {len(surface_stats)} "
          f"({', '.join(s['name'] for s in surface_stats[:5])}{'…' if len(surface_stats) > 5 else ''})")


if __name__ == "__main__":
    main()
