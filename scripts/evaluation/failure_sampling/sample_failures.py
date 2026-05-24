"""Sample failure cases per sub-category for qualitative error analysis.

For each analysis dimension (depth, object_coverage) this script finds
mispredicted samples per bin, randomly samples up to --n-samples of them,
and copies the frame images into a structured output directory.

When --dataset is provided, a second image with colored mask overlays is also
saved next to each plain frame:

    epic_kitchen  — hand mask (blue) + object mask (green)
                    + refined touch mask if available, else raw touch mask (red)
                    Touch mask is shown only for touch-positive samples.

    greatest_hits — stick mask (blue) + object mask (green)
                    + touch mask (red) for touch-positive samples.

Output layout
-------------
<output-dir>/
  depth_failures/
    near_distance/          (Close bin: disparity 170-255)
      sample_1.jpg
      sample_1_overlay.jpg  (present when --dataset is given)
      sample_2.jpg
      sample_2_overlay.jpg
      metadata.csv
    medium_distance/        (Medium bin: disparity 85-170)
      ...
    far_distance/           (Far bin: disparity 0-85)
      ...
  object_size_failures/
    small/                  (coverage < 0.02)
      ...
    medium_small/           (coverage 0.02-0.07)
      ...
    medium_large/           (coverage 0.07-0.15)
      ...
    large/                  (coverage >= 0.15)
      ...

Usage
-----
    python scripts/evaluation/failure_sampling/sample_failures.py \\
        results/predictions.csv \\
        --annotations /path/to/annotations.json \\
        --output-dir  results/evaluation/my_run \\
        [--dataset epic_kitchen] [--n-samples 3] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import enrich_df


# ── Bin definitions (must match the main analysis scripts) ────────────────────

_DEPTH_BINS = [
    # (dir_name, display_label, lo_inclusive, hi_exclusive)
    ("near_distance",   "Close",   170, 256),
    ("medium_distance", "Medium",   85, 170),
    ("far_distance",    "Far",       0,  85),
]

_COVERAGE_BINS = [
    ("small",        "Small",        0.00,  0.02),
    ("medium_small", "Medium-small", 0.02,  0.07),
    ("medium_large", "Medium-large", 0.07,  0.15),
    ("large",        "Large",        0.15,  1.01),
]

# Mask overlay colors: (R, G, B, alpha 0-255)
_COLOR_AGENT  = (50,  120, 255, 160)   # blue  — hand / stick
_COLOR_OBJECT = (50,  210,  60, 150)   # green — object
_COLOR_TOUCH  = (230,  45,  45, 170)   # red   — touch / refined touch


# ── Mask lookup ───────────────────────────────────────────────────────────────

def _mask_key(img_path: str, label: int) -> tuple[str, str, int]:
    """Stable lookup key: (video_dir_name, frame_filename, label).

    Using just basename collides for Greatest Hits where every video has
    frame_000001.jpg, frame_000002.jpg, etc.  Including the parent directory
    (which is always the video folder) disambiguates without depending on
    absolute path prefixes that differ across machines.
    """
    p = Path(img_path)
    return (p.parent.name, p.name, label)


def _build_mask_lookup(annotation_paths: list[Path]) -> dict[tuple[str, str, int], dict]:
    """Return {(video_dir, frame_filename, label_int): annotation_entry} from JSON files."""
    lookup: dict[tuple[str, str, int], dict] = {}
    for path in annotation_paths:
        if not path.exists():
            print(f"[WARN] annotation file not found: {path}")
            continue
        with open(path) as f:
            entries = json.load(f)
        for entry in entries:
            img = entry.get("image_path", "")
            if not img:
                continue
            label = 1 if entry.get("type") == "touch" else 0
            key = _mask_key(img, label)
            if key not in lookup:
                lookup[key] = entry
    return lookup


def _resolve_mask_paths(
    entry: dict,
    dataset: str,
    is_touch: bool,
) -> tuple[Path | None, Path | None, Path | None]:
    """Return (agent_mask, object_mask, touch_mask) Paths for one annotation entry.

    For epic_kitchen the touch mask used is the *refined* variant when it exists
    on disk, falling back to the raw touch mask.
    For greatest_hits the touch mask is target_path.
    In both cases touch_mask is None for no-touch samples.
    """
    def _p(key: str) -> Path | None:
        v = entry.get(key)
        return Path(v) if v else None

    if dataset == "epic_kitchen":
        agent = _p("hand_mask_path")
        obj   = _p("object_mask_path")
        if is_touch:
            raw = _p("touch_mask_path") or _p("target_path")
            if raw is not None:
                refined = raw.parent / (raw.stem + "_refined" + raw.suffix)
                touch = refined if refined.exists() else raw
            else:
                touch = None
        else:
            touch = None

    else:  # greatest_hits
        agent = _p("stick_mask_path")
        obj   = _p("object_mask_path")
        touch = _p("target_path") if is_touch else None

    return agent, obj, touch


# ── Overlay rendering ─────────────────────────────────────────────────────────

def _make_overlay(
    frame_path: Path,
    agent_mask: Path | None,
    object_mask: Path | None,
    touch_mask: Path | None,
    error_type: str,
) -> "PIL.Image.Image | None":
    """Composite masks onto the frame and return a PIL RGB image, or None on failure."""
    try:
        import numpy as np
        from PIL import Image, ImageDraw
    except ImportError:
        print("[WARN] Pillow not available — overlay images will be skipped")
        return None

    if not frame_path.exists():
        return None

    frame = Image.open(frame_path).convert("RGBA")
    W, H = frame.size

    layers = [
        (agent_mask,  _COLOR_AGENT,  "Agent"),
        (object_mask, _COLOR_OBJECT, "Object"),
        (touch_mask,  _COLOR_TOUCH,  "Touch"),
    ]

    legend_items: list[tuple[tuple[int, int, int], str]] = []
    for mask_path, color, label in layers:
        if mask_path is None or not mask_path.exists():
            continue
        mask_img = Image.open(mask_path).convert("L")
        if mask_img.size != (W, H):
            mask_img = mask_img.resize((W, H), Image.NEAREST)
        mask_arr = np.array(mask_img)
        overlay_arr = np.zeros((H, W, 4), dtype=np.uint8)
        overlay_arr[mask_arr > 0] = color
        frame = Image.alpha_composite(frame, Image.fromarray(overlay_arr, "RGBA"))
        legend_items.append((color[:3], label))

    result = frame.convert("RGB")
    if not legend_items:
        return result

    draw = ImageDraw.Draw(result)
    pad, sq, row_h = 8, 14, 18

    # Opaque dark background for the legend
    legend_h = len(legend_items) * row_h + pad
    draw.rectangle([0, 0, 112, legend_h + pad], fill=(15, 15, 15))

    y = pad
    for rgb, lbl in legend_items:
        draw.rectangle([pad, y, pad + sq, y + sq], fill=rgb)
        # Black shadow + white text for readability on any background
        draw.text((pad + sq + 5 + 1, y + 1), lbl, fill=(0, 0, 0))
        draw.text((pad + sq + 5,     y),     lbl, fill=(255, 255, 255))
        y += row_h

    # Error-type badge (bottom-left)
    badge_color = (200, 50, 50) if error_type == "FN" else (50, 100, 200)
    bx, by = pad, H - pad - 18
    draw.rectangle([bx - 2, by - 2, bx + 36, by + 16], fill=badge_color)
    draw.text((bx + 1, by + 1), error_type, fill=(0, 0, 0))
    draw.text((bx,     by),     error_type, fill=(255, 255, 255))

    return result


# ── Core sampling ─────────────────────────────────────────────────────────────

def _error_type(label: int, pred: int) -> str:
    if label == 1 and pred == 0:
        return "FN"
    if label == 0 and pred == 1:
        return "FP"
    return "correct"


def _sample_bin_failures(
    df: pd.DataFrame,
    col: str,
    lo: float,
    hi: float,
    n: int,
    seed: int,
    touch_only: bool,
) -> pd.DataFrame:
    mask = df[col].notna() & (df[col] >= lo) & (df[col] < hi)
    if touch_only:
        mask &= df["label"] == 1
    group = df[mask]
    failures = group[group["label"].astype(int) != group["prediction"].astype(int)]
    if failures.empty:
        return pd.DataFrame()
    return failures.sample(min(n, len(failures)), random_state=seed)


def _save_sample(
    row: pd.Series,
    dest_dir: Path,
    idx: int,
    mask_lookup: dict | None,
    dataset: str | None,
) -> dict:
    """Copy frame and (optionally) its overlay into dest_dir; return metadata."""
    src = Path(str(row["frame_path"]))
    ext = src.suffix or ".jpg"
    dst_name = f"sample_{idx + 1}{ext}"
    dst = dest_dir / dst_name

    err = _error_type(int(row["label"]), int(row["prediction"]))
    meta: dict = {
        "sample":      dst_name,
        "frame_path":  str(src),
        "label":       int(row["label"]),
        "prediction":  int(row["prediction"]),
        "error_type":  err,
    }
    for col in ("depth_touch", "object_coverage", "object_name"):
        if col in row.index and pd.notna(row[col]):
            meta[col] = row[col]

    if not src.exists():
        meta["missing_source"] = True
        return meta

    shutil.copy2(src, dst)

    if mask_lookup is not None and dataset is not None:
        key = _mask_key(str(src), int(row["label"]))
        entry = mask_lookup.get(key)
        if entry is not None:
            agent, obj, touch = _resolve_mask_paths(entry, dataset, int(row["label"]) == 1)
            overlay = _make_overlay(src, agent, obj, touch, err)
            if overlay is not None:
                overlay_name = f"sample_{idx + 1}_overlay{ext}"
                overlay.save(dest_dir / overlay_name)
                meta["overlay"] = overlay_name

    return meta


def _process_bins(
    df: pd.DataFrame,
    bins: list[tuple],
    col: str,
    out_root: Path,
    n_samples: int,
    base_seed: int,
    touch_only: bool,
    mask_lookup: dict | None,
    dataset: str | None,
) -> int:
    out_root.mkdir(parents=True, exist_ok=True)
    total = 0

    for i, (dir_name, display_label, lo, hi) in enumerate(bins):
        failures = _sample_bin_failures(df, col, lo, hi, n_samples, base_seed + i, touch_only)
        bin_dir = out_root / dir_name
        bin_dir.mkdir(exist_ok=True)

        if failures.empty:
            print(f"  [SKIP] {dir_name} ({display_label}): no failures in this bin")
            continue

        rows = [
            _save_sample(row, bin_dir, j, mask_lookup, dataset)
            for j, (_, row) in enumerate(failures.iterrows())
        ]
        pd.DataFrame(rows).to_csv(bin_dir / "metadata.csv", index=False)

        n = len(rows)
        total += n
        fp = sum(1 for r in rows if r["error_type"] == "FP")
        fn = sum(1 for r in rows if r["error_type"] == "FN")
        overlays = sum(1 for r in rows if "overlay" in r)
        detail = f"FP={fp} FN={fn}" if not touch_only else f"FN={fn}"
        ov_note = f", {overlays} overlays" if overlays else ""
        print(f"  {dir_name} ({display_label}): {n} sample(s) [{detail}{ov_note}]")

    return total


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("csv", type=Path, help="Predictions CSV")
    parser.add_argument("--annotations", nargs="+", type=Path, required=True,
                        help="Annotation JSON file(s)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Run output directory; depth_failures/ and "
                             "object_size_failures/ are created inside it")
    parser.add_argument("--dataset", choices=["epic_kitchen", "greatest_hits"], default=None,
                        help="Dataset type — enables mask overlay images alongside each sample. "
                             "epic_kitchen: hand + object + refined-touch masks. "
                             "greatest_hits: stick + object + touch masks.")
    parser.add_argument("--n-samples", type=int, default=3,
                        help="Max failures to sample per sub-category (default: 3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (default: 42)")
    parser.add_argument("--skip", nargs="*", choices=["depth", "coverage"], default=[],
                        metavar="DIM",
                        help="Dimensions to skip: depth  coverage")
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(args.csv)

    df = pd.read_csv(args.csv)
    df = enrich_df(df, args.annotations)

    mask_lookup: dict | None = None
    if args.dataset is not None:
        mask_lookup = _build_mask_lookup(args.annotations)
        print(f"Mask lookup: {len(mask_lookup)} entries (dataset={args.dataset})")

    skipped = set(args.skip or [])

    # ── Depth failures ────────────────────────────────────────────────────────
    if "depth" not in skipped:
        if "depth_touch" not in df.columns or df["depth_touch"].isna().all():
            print("[SKIP] depth failures — depth_touch not populated")
        else:
            print("\nDepth failures (touch-positive only — failures are FN):")
            total = _process_bins(
                df, _DEPTH_BINS, "depth_touch",
                args.output_dir / "depth_failures",
                args.n_samples, args.seed,
                touch_only=True,
                mask_lookup=mask_lookup,
                dataset=args.dataset,
            )
            print(f"  → {total} depth failure sample(s) saved")

    # ── Object size (coverage) failures ──────────────────────────────────────
    if "coverage" not in skipped:
        if "object_coverage" not in df.columns or df["object_coverage"].isna().all():
            print("[SKIP] object_size failures — object_coverage not populated")
        else:
            print("\nObject size failures (all samples — failures include FP and FN):")
            total = _process_bins(
                df, _COVERAGE_BINS, "object_coverage",
                args.output_dir / "object_size_failures",
                args.n_samples, args.seed + 100,
                touch_only=False,
                mask_lookup=mask_lookup,
                dataset=args.dataset,
            )
            print(f"  → {total} object_size failure sample(s) saved")


if __name__ == "__main__":
    main()
