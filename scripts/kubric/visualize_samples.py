"""
Run SegGPT + annotate_touch on a handful of Kubric samples and save visualizations.

For each sample a directory is created containing four images:
  context.jpg           — raw context frame
  context_masks.jpg     — context frame with GT object masks overlaid
  target.jpg            — raw target frame
  target_inferred.jpg   — target frame with inferred object masks + touch mask overlaid

Usage:
    python scripts/kubric/visualize_samples.py \\
        --data-root /scratch-shared/$USER \\
        --num-samples 10 \\
        --output-dir results/kubric_viz
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from inference_script.src.visualization import draw_dual_mask_viz


def _load_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _load_mask(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("L"))


def _make_label_map(mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
    label = np.zeros(mask1.shape, dtype=np.uint8)
    label[mask1 > 127] = 1
    label[mask2 > 127] = 2
    return label


def _predict_masks(
    target_pil: Image.Image,
    ctx_pil: Image.Image,
    ctx_label_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    from inference_script.src.seggpt import run_seggpt
    pred = run_seggpt(target_pil, ctx_pil, ctx_label_map, num_labels=2)
    return (pred == 1).astype(np.uint8) * 255, (pred == 2).astype(np.uint8) * 255


def _save(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr).save(path, quality=95)


def visualize_samples(
    data_root: Path,
    num_samples: int,
    output_dir: Path,
    seed: int,
    dilation: int,
    abs_d_threshold: float,
    local_radius: int,
) -> None:
    from touch_detection_alg.pipeline import annotate_touch

    anno_dir = data_root / "kubric" / "annotations"
    val_path = anno_dir / "val.json"
    ctx_path = anno_dir / "val_ctx_index.json"

    with open(val_path) as f:
        val_samples: list[dict] = json.load(f)
    with open(ctx_path) as f:
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

    rng = random.Random(seed)
    class_names = list(val_ctx_index.keys())
    class_names = rng.sample(class_names, min(num_samples, len(class_names)))

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for class_name in class_names:
        val_class_samples = [val_samples[i] for i in val_ctx_index[class_name]]

        train_indices = train_ctx_index.get(class_name, [])
        if train_indices:
            ctx = rng.choice([train_samples[i] for i in train_indices])
        else:
            if len(val_class_samples) <= 1:
                print(f"  Skipping {class_name!r}: not enough samples")
                continue
            ctx_idx = rng.randrange(len(val_class_samples))
            ctx = val_class_samples[ctx_idx]
            val_class_samples = [s for i, s in enumerate(val_class_samples) if i != ctx_idx]

        qry = rng.choice(val_class_samples)

        try:
            ctx_img = _load_rgb(ctx["image_path"])
            ctx_mask1 = _load_mask(ctx["object1_mask_path"])
            ctx_mask2 = _load_mask(ctx["object2_mask_path"])
            qry_img = _load_rgb(qry["image_path"])
        except (KeyError, FileNotFoundError, OSError) as exc:
            print(f"  Skipping {class_name!r}: load error — {exc}")
            continue

        ctx_label_map = _make_label_map(ctx_mask1, ctx_mask2)
        ctx_pil = Image.fromarray(ctx_img)

        ctx_touch_mask = annotate_touch(
            ctx_img,
            ctx_mask1,
            ctx_mask2,
            dilation=dilation,
            abs_d_threshold=abs_d_threshold,
            local_radius=local_radius,
        )

        pred_obj1, pred_obj2 = _predict_masks(Image.fromarray(qry_img), ctx_pil, ctx_label_map)

        touch_mask = annotate_touch(
            qry_img,
            pred_obj1,
            pred_obj2,
            dilation=dilation,
            abs_d_threshold=abs_d_threshold,
            local_radius=local_radius,
        )

        sample_dir = output_dir / f"sample_{saved:03d}_{class_name[:40]}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        _save(ctx_img, sample_dir / "context.jpg")
        _save(
            draw_dual_mask_viz(ctx_img, ctx_mask1, ctx_mask2, ctx_touch_mask, [], []),
            sample_dir / "context_masks.jpg",
        )
        _save(qry_img, sample_dir / "target.jpg")
        _save(
            draw_dual_mask_viz(qry_img, pred_obj1, pred_obj2, touch_mask, [], []),
            sample_dir / "target_inferred.jpg",
        )

        saved += 1
        print(f"  [{saved}/{num_samples}] {class_name!r} → {sample_dir.name}/")

    print(f"\nSaved {saved} sample(s) to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-root", type=Path, default=_ROOT / "data",
        help="Root directory under which the kubric/ dataset folder lives.",
    )
    parser.add_argument(
        "--num-samples", type=int, default=10, metavar="N",
        help="Number of samples to visualize.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=_ROOT / "results" / "kubric_viz",
        help="Directory where per-sample folders will be written.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dilation", type=int, default=10)
    parser.add_argument("--abs-d-threshold", type=float, default=0.05)
    parser.add_argument("--local-radius", type=int, default=10)
    args = parser.parse_args()

    visualize_samples(
        data_root=args.data_root,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        seed=args.seed,
        dilation=args.dilation,
        abs_d_threshold=args.abs_d_threshold,
        local_radius=args.local_radius,
    )


if __name__ == "__main__":
    main()
