"""
Assess the distribution of one or more annotation JSON files.

Usage
-----
    # Single file
    python -m scripts.epic_kitchen.annotation_distribution \
        --inputs data/epic_kitchen/annotations/train.json

    # Compare train vs val side-by-side
    python -m scripts.epic_kitchen.annotation_distribution \
        --inputs data/epic_kitchen/annotations/train.json \
                 data/epic_kitchen/annotations/val.json \
        --labels train val

    # Also check which entries have depth / refined masks on disk
    python -m scripts.epic_kitchen.annotation_distribution \
        --inputs data/epic_kitchen/annotations/train.json \
                 data/epic_kitchen/annotations/val.json \
        --labels train val \
        --check-masks

    # Show top-20 objects
    python -m scripts.epic_kitchen.annotation_distribution \
        --inputs data/epic_kitchen/annotations/train.json \
        --top-n 20
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


# ── helpers ───────────────────────────────────────────────────────────────────

def _load(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def _participant(entry: dict) -> str:
    return entry.get("video_id", "")[:3]


def _bar(frac: float, width: int = 30) -> str:
    filled = round(frac * width)
    return "█" * filled + "░" * (width - filled)


def _pct(n: int, total: int) -> str:
    return f"{100 * n / total:.1f}%" if total else "—"


def _depth_path(touch_path: str) -> Path:
    p = Path(touch_path)
    stem = p.stem.removesuffix("_touch")
    return p.parent / (stem + "_depth" + p.suffix)


def _refined_path(touch_path: str) -> Path:
    p = Path(touch_path)
    return p.parent / (p.stem + "_refined" + p.suffix)


def _valid(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


# ── mask status check ─────────────────────────────────────────────────────────

def mask_status(pairs: list[dict], label: str) -> None:
    total = len(pairs)

    n_depth_file    = 0
    n_refined_file  = 0
    n_depth_stat    = 0
    n_coverage_stat = 0
    n_xy_stat       = 0

    for p in pairs:
        touch = p.get("target_path") or p.get("touch_mask_path", "")
        if touch:
            if _valid(_depth_path(touch)):
                n_depth_file += 1
            if _valid(_refined_path(touch)):
                n_refined_file += 1
        if p.get("depth_touch") is not None:
            n_depth_stat += 1
        if p.get("object_coverage") is not None:
            n_coverage_stat += 1
        if p.get("x_touch") is not None and p.get("y_touch") is not None:
            n_xy_stat += 1

    n_missing_depth   = total - n_depth_file
    n_missing_refined = total - n_refined_file

    print(f"\n  mask & stats status  [{label}]")
    print(f"  {'─'*54}")
    print(f"  {'artifact':<30}  {'have':>7}  {'missing':>7}  {'%done':>7}")
    print(f"  {'─'*54}")

    def row(name, have):
        miss = total - have
        print(f"  {name:<30}  {have:>7,}  {miss:>7,}  {_pct(have, total):>7}")

    row("depth mask file (_depth.png)",  n_depth_file)
    row("refined mask  (_refined.png)", n_refined_file)
    row("depth_touch stat",             n_depth_stat)
    row("x_touch / y_touch stats",      n_xy_stat)
    row("object_coverage stat",         n_coverage_stat)

    print(f"\n  entries needing refinement (missing depth OR refined):")
    needs = sum(
        1 for p in pairs
        if not (
            _valid(_depth_path(p.get("target_path") or p.get("touch_mask_path", "")))
            and _valid(_refined_path(p.get("target_path") or p.get("touch_mask_path", "")))
        )
    )
    print(f"    {needs:,} / {total:,}  ({_pct(needs, total)} of entries)")


# ── per-file distribution report ──────────────────────────────────────────────

def report(pairs: list[dict], label: str, top_n: int, check_masks: bool) -> None:
    total        = len(pairs)
    n_touch      = sum(1 for p in pairs if p.get("type") == "touch")
    n_notouch    = total - n_touch
    frames       = {p["image_path"] for p in pairs}
    videos       = {p["video_id"]   for p in pairs}
    participants = {_participant(p) for p in pairs}

    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    print(f"  pairs        : {total:,}")
    print(f"  unique frames: {len(frames):,}")
    print(f"  videos       : {len(videos)}")
    print(f"  participants : {len(participants)}")

    touch_frac = n_touch / total if total else 0
    print(f"\n  type balance:")
    print(f"    touch    {_bar(touch_frac)} {n_touch:>6,}  {_pct(n_touch,   total):>6}")
    print(f"    no-touch {_bar(1-touch_frac)} {n_notouch:>6,}  {_pct(n_notouch, total):>6}")

    p_counts: Counter = Counter(_participant(p) for p in pairs)
    print(f"\n  per participant  ({len(p_counts)} total):")
    for pid, cnt in sorted(p_counts.items()):
        t = sum(1 for p in pairs if _participant(p) == pid and p.get("type") == "touch")
        print(f"    {pid}  {cnt:>6,} pairs  ({_pct(t, cnt)} touch)")

    v_counts: Counter = Counter(p["video_id"] for p in pairs)
    print(f"\n  per video  (top {min(top_n, len(v_counts))} of {len(v_counts)}):")
    for vid, cnt in v_counts.most_common(top_n):
        t = sum(1 for p in pairs if p["video_id"] == vid and p.get("type") == "touch")
        print(f"    {vid}  {cnt:>5,}  ({_pct(t, cnt)} touch)")

    obj_counts: Counter = Counter(
        p.get("object_name", "") or "(none)"
        for p in pairs
    )
    print(f"\n  top {min(top_n, len(obj_counts))} object classes  (of {len(obj_counts)} unique):")
    for name, cnt in obj_counts.most_common(top_n):
        bar = _bar(cnt / total, width=20)
        print(f"    {name:<30s} {bar} {cnt:>5,}  {_pct(cnt, total):>6}")

    coverages = [p["object_coverage"] for p in pairs
                 if p.get("object_coverage") is not None]
    if coverages:
        import statistics
        print(f"\n  object_coverage  (n={len(coverages):,}):")
        print(f"    min  {min(coverages):.4f}")
        print(f"    p25  {sorted(coverages)[len(coverages)//4]:.4f}")
        print(f"    med  {statistics.median(coverages):.4f}")
        print(f"    p75  {sorted(coverages)[3*len(coverages)//4]:.4f}")
        print(f"    max  {max(coverages):.4f}")

    if check_masks:
        mask_status(pairs, label)


# ── comparison table ───────────────────────────────────────────────────────────

def compare(all_pairs: list[list[dict]], labels: list[str], check_masks: bool) -> None:
    grand = sum(len(p) for p in all_pairs)

    print(f"\n{'─'*60}")
    print(f"  Cross-split comparison")
    print(f"{'─'*60}")
    header = f"  {'metric':<28}" + "".join(f"  {l:>12}" for l in labels)
    print(header)
    print(f"  {'─'*28}" + "".join("  " + "─"*12 for _ in labels))

    def row(name: str, vals: list) -> None:
        print(f"  {name:<28}" + "".join(f"  {str(v):>12}" for v in vals))

    row("pairs",         [f"{len(p):,}"                             for p in all_pairs])
    row("% of total",    [f"{_pct(len(p), grand)}"                  for p in all_pairs])
    row("unique frames", [f"{len({e['image_path'] for e in p}):,}"  for p in all_pairs])
    row("videos",        [f"{len({e['video_id'] for e in p})}"      for p in all_pairs])
    row("participants",  [f"{len({_participant(e) for e in p})}"    for p in all_pairs])
    row("touch",         [f"{sum(1 for e in p if e.get('type')=='touch'):,}" for p in all_pairs])
    row("no-touch",      [f"{sum(1 for e in p if e.get('type')!='touch'):,}" for p in all_pairs])
    row("touch %",       [f"{_pct(sum(1 for e in p if e.get('type')=='touch'), len(p))}" for p in all_pairs])

    if check_masks:
        def _have_depth(p):
            t = p.get("target_path") or p.get("touch_mask_path", "")
            return _valid(_depth_path(t)) if t else False
        def _have_refined(p):
            t = p.get("target_path") or p.get("touch_mask_path", "")
            return _valid(_refined_path(t)) if t else False

        row("depth mask",    [f"{sum(1 for e in p if _have_depth(e)):,}"   for p in all_pairs])
        row("refined mask",  [f"{sum(1 for e in p if _have_refined(e)):,}" for p in all_pairs])
        row("depth_touch",   [f"{sum(1 for e in p if e.get('depth_touch') is not None):,}" for p in all_pairs])
        row("coverage stat", [f"{sum(1 for e in p if e.get('object_coverage') is not None):,}" for p in all_pairs])


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Assess annotation JSON distribution",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="One or more annotation JSON files")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="Display labels matching --inputs (default: filename stems)")
    ap.add_argument("--top-n",  type=int, default=10,
                    help="Rows to show in per-video / object tables")
    ap.add_argument("--check-masks", action="store_true",
                    help="Check depth / refined mask files on disk and annotation stats completeness")
    args = ap.parse_args()

    paths  = [Path(i) for i in args.inputs]
    labels = args.labels or [p.stem for p in paths]

    if len(labels) != len(paths):
        ap.error("--labels count must match --inputs count")

    all_pairs = [_load(p) for p in paths]

    for pairs, label in zip(all_pairs, labels):
        report(pairs, label, args.top_n, args.check_masks)

    if len(all_pairs) > 1:
        compare(all_pairs, labels, args.check_masks)


if __name__ == "__main__":
    main()