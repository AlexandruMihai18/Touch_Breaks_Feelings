"""Callback that bins validation samples by GT touch-pixel coverage (quartiles)
and logs per-bin binary + segmentation metrics, plus one best/worst visualisation
per bin.

Coverage = fraction of pixels that are GT-positive in the touch mask.
Four quartile bins are recomputed each epoch from the observed coverage
distribution so each bin contains roughly equal sample counts regardless of
how sparse the touch annotations are.

WandB keys
----------
coverage/<Q1|Q2|Q3|Q4>/{accuracy,precision,recall,f1,iou,dice,n_samples}
viz/coverage_<bin>/best_gt_pred   | best_pred_mask
viz/coverage_<bin>/worst_gt_pred  | worst_pred_mask
"""

from __future__ import annotations

import math

import numpy as np
import torch
import wandb
from lightning import Callback, LightningModule, Trainer

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_BIN_NAMES = ("Q1", "Q2", "Q3", "Q4")


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def _denorm(img: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(_IMAGENET_MEAN, dtype=img.dtype)[:, None, None]
    std = torch.tensor(_IMAGENET_STD, dtype=img.dtype)[:, None, None]
    return (img * std + mean).clamp(0.0, 1.0)


def _cmy_overlay(
    img: torch.Tensor,
    gt_mask: torch.Tensor,
    pred_mask: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Cyan=GT-only, Yellow=Pred-only, Magenta=overlap."""
    out = img.clone()
    gt = gt_mask.bool()
    pred = pred_mask.bool()
    gt_only = gt & ~pred
    pred_only = pred & ~gt
    overlap = gt & pred
    boost = lambda c: alpha + (1.0 - alpha) * c  # noqa: E731
    dim = lambda c: (1.0 - alpha) * c  # noqa: E731
    out[0] = torch.where(gt_only, dim(img[0]), img[0])
    out[0] = torch.where(pred_only | overlap, boost(img[0]), out[0])
    out[1] = torch.where(gt_only | pred_only, boost(img[1]), img[1])
    out[1] = torch.where(overlap, dim(img[1]), out[1])
    out[2] = torch.where(gt_only | overlap, boost(img[2]), img[2])
    out[2] = torch.where(pred_only, dim(img[2]), out[2])
    return out.clamp(0.0, 1.0)


def _render_overlay(entry: dict) -> np.ndarray:
    img = _denorm(entry["pixel_values"])
    overlay = _cmy_overlay(img, entry["gt_bin"].bool(), entry["pred_bin"])
    return (overlay.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _render_pred_mask(entry: dict) -> np.ndarray:
    """White = predicted positive; red = GT touch on top."""
    pred = entry["pred_bin"]
    gt = entry["gt_bin"].bool()
    canvas = (
        (pred.float() * 255)
        .clamp(0, 255)
        .byte()
        .unsqueeze(-1)
        .expand(-1, -1, 3)
        .clone()
    )
    canvas[gt] = torch.tensor([255, 0, 0], dtype=torch.uint8)
    return canvas.numpy()


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _safe_div(n: float, d: float) -> float:
    return n / d if d > 0 else float("nan")


def _per_sample_dice(pred: torch.Tensor, gt: torch.Tensor) -> float:
    p = pred.view(-1).float()
    t = gt.view(-1).float()
    return (2.0 * (p * t).sum() / (p.sum() + t.sum() + 1e-6)).item()


def _bin_metrics(samples: list[dict]) -> dict[str, float]:
    """Aggregate pixel-level TP/FP/FN/TN across all samples in a bin."""
    tp = fp = fn = tn = 0
    for s in samples:
        p = s["pred_bin"].bool()
        g = s["gt_bin"].bool()
        tp += int((p & g).sum())
        fp += int((p & ~g).sum())
        fn += int((~p & g).sum())
        tn += int((~p & ~g).sum())
    n = tp + fp + fn + tn
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    if not math.isnan(precision) and not math.isnan(recall) and precision + recall > 0:
        f1 = _safe_div(2 * precision * recall, precision + recall)
    else:
        f1 = float("nan")
    return {
        "accuracy": _safe_div(tp + tn, n),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": _safe_div(tp, tp + fp + fn),
        "dice": _safe_div(2 * tp, 2 * tp + fp + fn),
        "n_samples": float(len(samples)),
    }


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------


class CoverageBinMetricsCallback(Callback):
    """Bins validation samples by GT touch coverage quartiles and logs
    per-bin binary + segmentation metrics plus one best/worst visualisation
    per bin (ranked by per-sample Dice score).
    """

    def __init__(self) -> None:
        self._reset()

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx,
        dataloader_idx: int = 0,
    ) -> None:
        if not isinstance(outputs, dict):
            return
        preds = outputs.get("preds")
        gt = outputs.get("gt")
        if preds is None or gt is None:
            return

        # .cpu() before dtype conversion to avoid a transient float GPU tensor.
        preds = preds.detach().cpu().float()
        gt = gt.detach().cpu()
        # qry_touch_bin is the raw dataset binary mask — used for coverage and viz.
        gt_bin_batch = batch["qry_touch_bin"].cpu()
        pixel_values = batch["pixel_values"].cpu()

        for i in range(preds.shape[0]):
            # Store masks as bool: PyTorch bool is uint8 (1 byte/pixel vs 4 for
            # float32), so this cuts per-sample mask memory by 4x.  All
            # consumers either call .bool() themselves or call .float() first,
            # so the dtype change is transparent.
            pred_bool = preds[i].bool()
            gt_bool = gt_bin_batch[i].bool()
            gt_metric_bool = gt[i].bool()
            self._samples.append(
                {
                    "pixel_values": pixel_values[i],    # float32 needed for viz
                    "gt_bin": gt_bool,                  # for viz and coverage
                    "pred_bin": pred_bool,
                    "gt_metric": gt_metric_bool,        # from _binarise(), for metrics
                    "coverage": gt_bool.float().mean().item(),
                    "dice": _per_sample_dice(pred_bool, gt_metric_bool),
                }
            )

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if trainer.sanity_checking or not self._samples:
            self._reset()
            return

        coverages = np.array([s["coverage"] for s in self._samples])
        q25, q50, q75 = (
            float(np.percentile(coverages, 25)),
            float(np.percentile(coverages, 50)),
            float(np.percentile(coverages, 75)),
        )

        bins: list[list[dict]] = [[], [], [], []]
        for s in self._samples:
            c = s["coverage"]
            if c < q25:
                bins[0].append(s)
            elif c < q50:
                bins[1].append(s)
            elif c < q75:
                bins[2].append(s)
            else:
                bins[3].append(s)

        if trainer.logger is None or wandb.run is None:
            self._reset()
            return

        bin_labels = [
            f"cov<{q25:.4f}",
            f"{q25:.4f}≤cov<{q50:.4f}",
            f"{q50:.4f}≤cov<{q75:.4f}",
            f"cov≥{q75:.4f}",
        ]

        log_dict: dict = {}
        for name, samples, label in zip(_BIN_NAMES, bins, bin_labels):
            if not samples:
                continue

            # Override gt_bin with gt_metric for metric computation only
            metric_samples = [
                {**s, "gt_bin": s["gt_metric"]} for s in samples
            ]
            metrics = _bin_metrics(metric_samples)
            for metric, value in metrics.items():
                if not isinstance(value, float) or not math.isnan(value):
                    log_dict[f"coverage_{name}/{metric}"] = value

            ranked = sorted(samples, key=lambda s: s["dice"])
            worst, best = ranked[0], ranked[-1]

            for tag, entry in (("best", best), ("worst", worst)):
                caption = (
                    f"{tag} | dice={entry['dice']:.3f}"
                    f" | cov={entry['coverage']:.4f} | {label}"
                )
                log_dict[f"coverage_{name}/viz_{tag}_gt_pred"] = [
                    wandb.Image(_render_overlay(entry), caption=caption)
                ]
                log_dict[f"coverage_{name}/viz_{tag}_pred_mask"] = [
                    wandb.Image(_render_pred_mask(entry), caption=caption)
                ]

        trainer.logger.experiment.log(log_dict)
        self._reset()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def _reset(self) -> None:
        self._samples: list[dict] = []
