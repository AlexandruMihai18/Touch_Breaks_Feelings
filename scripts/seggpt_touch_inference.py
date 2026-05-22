"""
Evaluate the SegGPT + annotate_touch pipeline on Epic Kitchen or Greatest Hits.

For each class in {split}_ctx_index.json:
  1. Pick one random context frame from the class.
  2. Build a 2-class prompt mask from the context ground truth: agent=1, object=2.
  3. Run SegGPT on every other frame in the class to predict agent/object masks.
  4. Run annotate_touch on the predicted masks to detect the touch region.
  5. Predict touch=1 if touch area > 0, else 0.

Writes results/{dataset}_{split}_seggpt_touch_results.csv with binary metrics.
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from inference_script.src.seggpt import run_seggpt
from touch_detection_alg.pipeline import annotate_touch

_DATASETS = {
    "epic_kitchen": {
        "anno_dir": _ROOT / "data" / "epic_kitchen" / "annotations",
        "ctx_key": "object_name",
        "agent_mask_key": "hand_mask_path",
    },
    "greatest_hits": {
        "anno_dir": _ROOT / "data" / "greatest_hits" / "annotations",
        "ctx_key": "video_id",
        "agent_mask_key": "stick_mask_path",
    },
}

_CSV_FIELDS = [
    "frame_id", "video_id", "frame_path", "audio_path",
    "label", "prediction", "x_touch", "y_touch",
]


def _load_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _load_mask(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("L"))


def _frame_id(image_path: str) -> str:
    return Path(image_path).stem


def _make_label_map(agent_mask: np.ndarray, obj_mask: np.ndarray) -> np.ndarray:
    """2D label map: 0=background, 1=agent (hand/stick), 2=object."""
    label = np.zeros(agent_mask.shape, dtype=np.uint8)
    label[agent_mask > 127] = 1
    label[obj_mask > 127] = 2
    return label


def _predict_masks(
    target_pil: Image.Image,
    ctx_pil: Image.Image,
    ctx_label_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run SegGPT with a 2-class prompt. Returns (agent_mask, obj_mask) as uint8 [0,255]."""
    pred = run_seggpt(target_pil, ctx_pil, ctx_label_map, num_labels=2)
    return (pred == 1).astype(np.uint8) * 255, (pred == 2).astype(np.uint8) * 255


def run_inference(
    dataset: str,
    split: str,
    seed: int,
    max_classes: int | None,
    dilation: int,
    abs_d_threshold: float,
    local_radius: int,
) -> list[dict]:
    cfg = _DATASETS[dataset]
    anno_dir: Path = cfg["anno_dir"]
    agent_key: str = cfg["agent_mask_key"]

    with open(anno_dir / f"{split}.json") as f:
        samples: list[dict] = json.load(f)
    with open(anno_dir / f"{split}_ctx_index.json") as f:
        ctx_index: dict[str, list[int]] = json.load(f)

    rng = random.Random(seed)
    class_names = list(ctx_index.keys())
    if max_classes is not None:
        class_names = rng.sample(class_names, min(max_classes, len(class_names)))

    results: list[dict] = []
    n_classes = len(class_names)

    for cls_i, class_name in enumerate(class_names, 1):
        class_samples = [samples[i] for i in ctx_index[class_name]]
        if len(class_samples) < 2:
            print(f"[{cls_i}/{n_classes}] {class_name!r}: skipped (only {len(class_samples)} sample)")
            continue

        ctx_idx = rng.randrange(len(class_samples))
        ctx = class_samples[ctx_idx]
        queries = [s for i, s in enumerate(class_samples) if i != ctx_idx]

        try:
            ctx_img = _load_rgb(ctx["image_path"])
            ctx_agent = _load_mask(ctx[agent_key])
            ctx_obj = _load_mask(ctx["object_mask_path"])
        except (FileNotFoundError, OSError) as exc:
            print(f"[{cls_i}/{n_classes}] {class_name!r}: context load error — {exc}")
            continue

        ctx_label_map = _make_label_map(ctx_agent, ctx_obj)
        ctx_pil = Image.fromarray(ctx_img)

        print(f"[{cls_i}/{n_classes}] {class_name!r}: {len(queries)} query frames", flush=True)

        for q_i, qry in enumerate(queries, 1):
            try:
                qry_img = _load_rgb(qry["image_path"])
            except (FileNotFoundError, OSError) as exc:
                print(f"  [{q_i}/{len(queries)}] load error — {exc}")
                continue

            pred_agent, pred_obj = _predict_masks(Image.fromarray(qry_img), ctx_pil, ctx_label_map)

            touch_mask = annotate_touch(
                qry_img,
                pred_agent,
                pred_obj,
                dilation=dilation,
                abs_d_threshold=abs_d_threshold,
                local_radius=local_radius,
            )

            label = 1 if qry.get("type") == "touch" else 0
            prediction = 1 if touch_mask.any() else 0

            results.append({
                "frame_id": _frame_id(qry["image_path"]),
                "video_id": qry["video_id"],
                "frame_path": qry["image_path"],
                "audio_path": qry.get("audio_path"),
                "label": label,
                "prediction": prediction,
                "x_touch": None,
                "y_touch": None,
            })

            if q_i % 10 == 0 or q_i == len(queries):
                print(f"  {q_i}/{len(queries)} done", flush=True)

    return results


def _print_metrics(results: list[dict]) -> None:
    labels = [r["label"] for r in results]
    preds = [r["prediction"] for r in results]
    tp = sum(l == 1 and p == 1 for l, p in zip(labels, preds))
    tn = sum(l == 0 and p == 0 for l, p in zip(labels, preds))
    fp = sum(l == 0 and p == 1 for l, p in zip(labels, preds))
    fn = sum(l == 1 and p == 0 for l, p in zip(labels, preds))
    n = len(results)
    acc = (tp + tn) / n
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    sep = "─" * 40
    print(f"\n{sep}")
    print(f"  Samples   : {n}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=list(_DATASETS), help="Dataset to evaluate.")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for context sampling.")
    parser.add_argument(
        "--max-classes", type=int, default=None, metavar="N",
        help="Limit to N random classes (useful for quick smoke-tests).",
    )
    parser.add_argument("--dilation", type=int, default=10, help="Contact zone dilation radius (px).")
    parser.add_argument("--abs-d-threshold", type=float, default=0.05, help="Max |depth1-depth2| to keep a contact pixel.")
    parser.add_argument("--local-radius", type=int, default=10, help="Box-window half-width for per-pixel depth estimates.")
    parser.add_argument(
        "--output-dir", type=Path, default=_ROOT / "results",
        help="Directory for the output CSV (created if missing).",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = run_inference(
        dataset=args.dataset,
        split=args.split,
        seed=args.seed,
        max_classes=args.max_classes,
        dilation=args.dilation,
        abs_d_threshold=args.abs_d_threshold,
        local_radius=args.local_radius,
    )

    out_csv = args.output_dir / f"{args.dataset}_{args.split}_seggpt_touch_results.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nWrote {len(results)} rows → {out_csv}")
    if results:
        _print_metrics(results)


if __name__ == "__main__":
    main()
