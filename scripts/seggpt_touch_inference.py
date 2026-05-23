"""
Evaluate the SegGPT + annotate_touch pipeline on Epic Kitchen or Greatest Hits.

For each class in val_ctx_index.json:
  1. Pick one random context frame from the class.
  2. Build a 2-class prompt mask from the context ground truth: agent=1, object=2.
  3. Run SegGPT on every other frame in the class to predict agent/object masks.
  4. Run annotate_touch on the predicted masks to detect the touch region.
  5. Predict touch=1 if touch area > 0, else 0.

Writes results/{dataset}_val_seggpt_touch_results.csv with binary metrics.
"""

import argparse
import csv
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

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
    "kubric_movi_a_256": {
        "ctx_key": "object_name",
        "mask1_key": "object1_mask_path",
        "mask2_key": "object2_mask_path",
    },
}

_CSV_FIELDS = [
    "frame_id", "video_id", "frame_path", "audio_path",
    "label", "prediction", "x_touch", "y_touch", "iou",
]


def _load_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _load_mask(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("L"))


def _frame_id(image_path: str) -> str:
    return Path(image_path).stem


def _compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Pixel-level Jaccard IoU between two uint8 masks. Returns 1.0 when both are empty."""
    pred_bool = pred > 127
    gt_bool = gt > 127
    intersection = (pred_bool & gt_bool).sum()
    union = (pred_bool | gt_bool).sum()
    return float(intersection / union) if union > 0 else 1.0


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
    from inference_script.src.seggpt import run_seggpt

    pred = run_seggpt(target_pil, ctx_pil, ctx_label_map, num_labels=2)
    return (pred == 1).astype(np.uint8) * 255, (pred == 2).astype(np.uint8) * 255


def run_inference(
    dataset: str,
    data_root: Path,
    seed: int,
    max_classes: int | None,
    dilation: int,
    abs_d_threshold: float,
    local_radius: int,
    log: logging.Logger,
    num_jobs: int = 1,
    job_index: int = 0,
) -> list[dict]:
    from touch_detection_alg.pipeline import annotate_touch

    cfg = _DATASETS[dataset]
    anno_dir: Path = data_root / dataset / "annotations"
    mask1_key: str = cfg["mask1_key"]
    mask2_key: str = cfg["mask2_key"]

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

    rng = random.Random(seed)
    class_names = list(val_ctx_index.keys())
    if max_classes is not None:
        class_names = rng.sample(class_names, min(max_classes, len(class_names)))

    # Round-robin slice across array tasks; all jobs compute the same class_names
    # first (same seed) so each gets a deterministic, non-overlapping subset.
    class_names = class_names[job_index::num_jobs]

    shard_info = f" [job {job_index + 1}/{num_jobs}]" if num_jobs > 1 else ""
    log.info("Running val inference on %d classes%s  seed=%d", len(class_names), shard_info, seed)

    results: list[dict] = []
    n_classes = len(class_names)
    stats = {"train_ctx": 0, "val_ctx": 0, "skipped": 0}

    for cls_i, class_name in enumerate(class_names, 1):
        val_class_samples = [val_samples[i] for i in val_ctx_index[class_name]]

        # Prefer a context frame from the training distribution.
        train_indices = train_ctx_index.get(class_name, [])
        if train_indices:
            train_class_samples = [train_samples[i] for i in train_indices]
            ctx = rng.choice(train_class_samples)
            queries = val_class_samples
            ctx_source = f"train ({len(train_indices)} candidates)"
            stats["train_ctx"] += 1
        else:
            # No train samples for this class — fall back to val.
            # Skip if only one val sample to avoid using it as both context and query.
            if len(val_class_samples) <= 1:
                log.warning(
                    "[%d/%d] %r: skipped — no train context, only %d val sample(s)",
                    cls_i, n_classes, class_name, len(val_class_samples),
                )
                stats["skipped"] += 1
                continue
            ctx_idx = rng.randrange(len(val_class_samples))
            ctx = val_class_samples[ctx_idx]
            queries = [s for i, s in enumerate(val_class_samples) if i != ctx_idx]
            ctx_source = f"val fallback ({len(val_class_samples)} val samples)"
            stats["val_ctx"] += 1

        try:
            ctx_img = _load_rgb(ctx["image_path"])
            ctx_mask1 = _load_mask(ctx[mask1_key])
            ctx_mask2 = _load_mask(ctx[mask2_key])
        except (KeyError, FileNotFoundError, OSError) as exc:
            log.error("[%d/%d] %r: context load error — %s", cls_i, n_classes, class_name, exc)
            stats["skipped"] += 1
            continue

        ctx_label_map = _make_label_map(ctx_mask1, ctx_mask2)
        ctx_pil = Image.fromarray(ctx_img)

        log.info(
            "[%d/%d] %r  ctx=%s  queries=%d  ctx_frame=%s",
            cls_i, n_classes, class_name, ctx_source, len(queries),
            Path(ctx["image_path"]).name,
        )

        for q_i, qry in enumerate(queries, 1):
            try:
                qry_img = _load_rgb(qry["image_path"])
            except (FileNotFoundError, OSError) as exc:
                log.warning("  [%d/%d] load error — %s", q_i, len(queries), exc)
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

            iou: float | None = None
            gt_path = qry.get("target_path")
            if gt_path:
                try:
                    gt_touch = _load_mask(gt_path)
                    iou = _compute_iou(touch_mask, gt_touch)
                except (FileNotFoundError, OSError):
                    pass

            results.append({
                "frame_id": _frame_id(qry["image_path"]),
                "video_id": qry["video_id"],
                "frame_path": qry["image_path"],
                "audio_path": qry.get("audio_path"),
                "label": label,
                "prediction": prediction,
                "x_touch": None,
                "y_touch": None,
                "iou": iou,
            })

            if q_i % 10 == 0 or q_i == len(queries):
                log.info("  %d/%d done", q_i, len(queries))

    log.info(
        "Context sourcing — train: %d  val-fallback: %d  skipped: %d",
        stats["train_ctx"], stats["val_ctx"], stats["skipped"],
    )
    return results


def _log_metrics(results: list[dict], log: logging.Logger) -> None:
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
    iou_vals = [r["iou"] for r in results if r.get("iou") is not None]
    mean_iou = sum(iou_vals) / len(iou_vals) if iou_vals else None

    sep = "─" * 40
    log.info(sep)
    log.info("  Samples   : %d", n)
    log.info("  Accuracy  : %.4f", acc)
    log.info("  Precision : %.4f", prec)
    log.info("  Recall    : %.4f", rec)
    log.info("  F1        : %.4f", f1)
    if mean_iou is not None:
        log.info("  Mean IoU  : %.4f  (n=%d)", mean_iou, len(iou_vals))
    log.info("  TP=%d  TN=%d  FP=%d  FN=%d", tp, tn, fp, fn)
    log.info(sep)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=list(_DATASETS), help="Dataset to evaluate.")
    parser.add_argument(
        "--data-root", type=Path, default=_ROOT / "data",
        help="Root directory under which dataset folders (epic_kitchen/, greatest_hits/) live.",
    )
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
    parser.add_argument("--num-jobs",  type=int, default=1,
                        help="Total number of parallel SLURM array tasks.")
    parser.add_argument("--job-index", type=int, default=0,
                        help="0-based index of this task (SLURM_ARRAY_TASK_ID).")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{args.dataset}_val_seggpt_touch_results"
    suffix = f"_{args.job_index}" if args.num_jobs > 1 else ""

    log_path = args.output_dir / f"{stem}{suffix}.log"
    log = logging.getLogger("seggpt_inference")
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    log.addHandler(sh)
    log.addHandler(fh)

    log.info("dataset=%s  data_root=%s  seed=%d", args.dataset, args.data_root, args.seed)

    results = run_inference(
        dataset=args.dataset,
        data_root=args.data_root,
        seed=args.seed,
        max_classes=args.max_classes,
        dilation=args.dilation,
        abs_d_threshold=args.abs_d_threshold,
        local_radius=args.local_radius,
        log=log,
        num_jobs=args.num_jobs,
        job_index=args.job_index,
    )

    out_csv = args.output_dir / f"{stem}{suffix}.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    shard_label = f" (shard {args.job_index + 1}/{args.num_jobs})" if args.num_jobs > 1 else ""
    log.info("Wrote %d rows → %s%s", len(results), out_csv, shard_label)
    log.info("Log saved → %s", log_path)
    if results:
        _log_metrics(results, log)


if __name__ == "__main__":
    main()
