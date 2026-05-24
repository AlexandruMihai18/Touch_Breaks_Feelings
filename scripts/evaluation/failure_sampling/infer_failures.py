"""
Run SegGPT + annotate_touch on failure cases sampled by sample_failures.py.

For each bin directory in the failure case output structure, reads metadata.csv,
looks up a context frame from the ctx_index, runs SegGPT inference on each
sample, and saves inferred masks alongside the originals.

Output added to each bin directory:
  sample_N_ctx.jpg         — raw context frame used for inference
  sample_N_ctx_masks.jpg   — context frame with GT agent/object/touch masks overlaid
  sample_N_agent_mask.png  — inferred agent/hand mask
  sample_N_object_mask.png — inferred object mask
  sample_N_touch_mask.png  — inferred touch mask
  sample_N_inferred.jpg    — frame with all inferred masks overlaid

Usage:
    python scripts/evaluation/failure_sampling/infer_failures.py \\
        results/evaluation/my_run \\
        --dataset epic_kitchen \\
        --data-root /scratch-shared/$USER
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

from inference_script.src.visualization import draw_dual_mask_viz

_DATASETS = {
    "epic_kitchen": {
        "ctx_key": "object_name",
        "mask1_key": "hand_mask_path",
        "mask2_key": "object_mask_path",
    },
    "greatest_hits": {
        "ctx_key": "video_id",
        "mask1_key": "stick_mask_path",
        "mask2_key": "object_mask_path",
    },
    "kubric": {
        "ctx_key": "object_name",
        "mask1_key": "object1_mask_path",
        "mask2_key": "object2_mask_path",
    },
}


def _load_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _load_mask(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("L"))


def _make_label_map(mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
    label = np.zeros(mask1.shape, dtype=np.uint8)
    label[mask1 > 127] = 1
    label[mask2 > 127] = 2
    return label


def _save_jpg(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr).save(path, quality=95)


def _load_annotations(
    anno_dir: Path,
) -> tuple[list[dict], dict[str, list[int]], list[dict], dict[str, list[int]]]:
    with open(anno_dir / "val.json") as f:
        val_samples: list[dict] = json.load(f)
    with open(anno_dir / "val_ctx_index.json") as f:
        val_ctx_index: dict[str, list[int]] = json.load(f)

    train_path = anno_dir / "train.json"
    train_ctx_path = anno_dir / "train_ctx_index.json"
    if train_path.exists() and train_ctx_path.exists():
        with open(train_path) as f:
            train_samples: list[dict] = json.load(f)
        with open(train_ctx_path) as f:
            train_ctx_index: dict[str, list[int]] = json.load(f)
    else:
        train_samples = []
        train_ctx_index = {}

    return val_samples, val_ctx_index, train_samples, train_ctx_index


def _ctx_key_for_row(row: pd.Series, dataset: str) -> str | None:
    """Return the ctx_index lookup key for this metadata row."""
    if _DATASETS[dataset]["ctx_key"] == "object_name":
        val = row.get("object_name")
        return str(val) if pd.notna(val) else None
    # greatest_hits: video_id is derived from the frame parent directory
    fp = row.get("frame_path")
    return Path(str(fp)).parent.name if pd.notna(fp) else None


def _pick_context(
    class_name: str,
    val_samples: list[dict],
    val_ctx_index: dict[str, list[int]],
    train_samples: list[dict],
    train_ctx_index: dict[str, list[int]],
    rng: random.Random,
    exclude_frame: str | None = None,
) -> dict | None:
    """Pick a context annotation entry, preferring train samples."""
    train_indices = train_ctx_index.get(class_name, [])
    if train_indices:
        return rng.choice([train_samples[i] for i in train_indices])

    val_indices = val_ctx_index.get(class_name, [])
    if not val_indices:
        return None

    candidates = [val_samples[i] for i in val_indices]
    if exclude_frame is not None:
        candidates = [s for s in candidates if s.get("image_path") != exclude_frame]
    return rng.choice(candidates) if candidates else None


def _find_bin_dirs(failure_dir: Path) -> list[Path]:
    """Return all bin directories with a metadata.csv under depth/object_size failures."""
    found: list[Path] = []
    for top in ("depth_failures", "object_size_failures"):
        top_dir = failure_dir / top
        if not top_dir.exists():
            continue
        for csv_path in sorted(top_dir.rglob("metadata.csv")):
            found.append(csv_path.parent)
    return found


def process_bin(
    bin_dir: Path,
    dataset: str,
    val_samples: list[dict],
    val_ctx_index: dict[str, list[int]],
    train_samples: list[dict],
    train_ctx_index: dict[str, list[int]],
    rng: random.Random,
    dilation: int,
    abs_d_threshold: float,
    local_radius: int,
) -> int:
    from touch_detection_alg.pipeline import annotate_touch
    from inference_script.src.seggpt import run_seggpt

    mask1_key = _DATASETS[dataset]["mask1_key"]
    mask2_key = _DATASETS[dataset]["mask2_key"]

    df = pd.read_csv(bin_dir / "metadata.csv")
    print(f"  {bin_dir.name}: {len(df)} sample(s)")

    # Cache one context per class_name to avoid re-selecting on every row
    ctx_cache: dict[str, dict | None] = {}
    done = 0

    for _, row in df.iterrows():
        sample_stem = Path(str(row["sample"])).stem
        inferred_path = bin_dir / f"{sample_stem}_inferred.jpg"
        if inferred_path.exists():
            print(f"    {sample_stem}: already inferred — skip")
            continue

        frame_path = str(row.get("frame_path", ""))
        if not frame_path or not Path(frame_path).exists():
            print(f"    {sample_stem}: frame not found — {frame_path}")
            continue

        class_name = _ctx_key_for_row(row, dataset)
        if class_name is None:
            print(f"    {sample_stem}: no ctx key — skip")
            continue

        if class_name not in ctx_cache:
            ctx_cache[class_name] = _pick_context(
                class_name,
                val_samples, val_ctx_index,
                train_samples, train_ctx_index,
                rng,
                exclude_frame=frame_path,
            )

        ctx = ctx_cache[class_name]
        if ctx is None:
            print(f"    {sample_stem}: no context for {class_name!r} — skip")
            continue

        try:
            ctx_img = _load_rgb(ctx["image_path"])
            ctx_mask1 = _load_mask(ctx[mask1_key])
            ctx_mask2 = _load_mask(ctx[mask2_key])
        except (KeyError, FileNotFoundError, OSError) as exc:
            print(f"    {sample_stem}: context load error — {exc}")
            ctx_cache[class_name] = None
            continue

        try:
            qry_img = _load_rgb(frame_path)
        except (FileNotFoundError, OSError) as exc:
            print(f"    {sample_stem}: query load error — {exc}")
            continue

        ctx_label_map = _make_label_map(ctx_mask1, ctx_mask2)

        ctx_touch_mask = annotate_touch(
            ctx_img, ctx_mask1, ctx_mask2,
            dilation=dilation,
            abs_d_threshold=abs_d_threshold,
            local_radius=local_radius,
        )
        _save_jpg(ctx_img, bin_dir / f"{sample_stem}_ctx.jpg")
        _save_jpg(
            draw_dual_mask_viz(ctx_img, ctx_mask1, ctx_mask2, ctx_touch_mask, [], []),
            bin_dir / f"{sample_stem}_ctx_masks.jpg",
        )

        pred = run_seggpt(
            Image.fromarray(qry_img),
            Image.fromarray(ctx_img),
            ctx_label_map,
            num_labels=2,
        )
        pred_agent = (pred == 1).astype(np.uint8) * 255
        pred_obj = (pred == 2).astype(np.uint8) * 255

        touch_mask = annotate_touch(
            qry_img, pred_agent, pred_obj,
            dilation=dilation,
            abs_d_threshold=abs_d_threshold,
            local_radius=local_radius,
        )

        Image.fromarray(pred_agent).save(bin_dir / f"{sample_stem}_agent_mask.png")
        Image.fromarray(pred_obj).save(bin_dir / f"{sample_stem}_object_mask.png")
        Image.fromarray(touch_mask).save(bin_dir / f"{sample_stem}_touch_mask.png")
        _save_jpg(
            draw_dual_mask_viz(qry_img, pred_agent, pred_obj, touch_mask, [], []),
            inferred_path,
        )
        touch_flag = "touch" if touch_mask.any() else "no-touch"
        print(f"    {sample_stem}: done [{touch_flag}]")
        done += 1

    return done


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "dirs", nargs="+", type=Path,
        help="One or more failure_sampling output directories.",
    )
    parser.add_argument(
        "--dataset", choices=list(_DATASETS), required=True,
        help="Dataset type — determines mask keys and ctx_index lookup.",
    )
    parser.add_argument(
        "--data-root", type=Path, default=_ROOT / "data",
        help="Root under which {dataset}/annotations/ lives.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dilation", type=int, default=10)
    parser.add_argument("--abs-d-threshold", type=float, default=0.05)
    parser.add_argument("--local-radius", type=int, default=10)
    args = parser.parse_args()

    anno_dir = args.data_root / args.dataset / "annotations"
    val_samples, val_ctx_index, train_samples, train_ctx_index = _load_annotations(anno_dir)
    print(f"Loaded {len(val_samples)} val, {len(train_samples)} train samples")
    print(f"Val classes: {len(val_ctx_index)}  Train classes: {len(train_ctx_index)}")

    rng = random.Random(args.seed)
    total_bins = total_done = 0

    for d in args.dirs:
        if not d.exists():
            print(f"[WARN] not found: {d}")
            continue
        bin_dirs = _find_bin_dirs(d)
        if not bin_dirs:
            print(f"[WARN] no bin dirs with metadata.csv in {d}")
            continue
        print(f"\n{d}  ({len(bin_dirs)} bin(s))")
        for bin_dir in bin_dirs:
            n = process_bin(
                bin_dir, args.dataset,
                val_samples, val_ctx_index,
                train_samples, train_ctx_index,
                rng, args.dilation, args.abs_d_threshold, args.local_radius,
            )
            total_bins += 1
            total_done += n

    print(f"\nDone — {total_done} sample(s) inferred across {total_bins} bin(s)")


if __name__ == "__main__":
    main()
