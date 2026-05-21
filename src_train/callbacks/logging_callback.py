import numpy as np
import torch
from lightning import Callback
from lightning.fabric.utilities.rank_zero import rank_zero_warn

import wandb

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# Black (0,0,0) normalised with ImageNet stats — background reference for the
# mask-integrity check.
_BG_NORM = (-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225)  # ≈ (-2.12, -2.04, -1.80)


class SegGPTLoggingCallback(Callback):
    """Handles all visual logging and debugging diagnostics for SegGPT training.

    Consumes the dict returned by SegGPTTouchModule.validation_step and fires
    WandB galleries each epoch:

      viz/gt_pred_compare   — CMY overlay on image: Cyan=GT-only, Yellow=Pred-only, Magenta=overlap
      viz/pred_mask         — binary predicted mask (white) with GT touch drawn in red on top
      viz/best_gt_pred      — highest-Dice samples, CMY overlay
      viz/best_pred_mask    — highest-Dice samples, predicted mask
      viz/worst_gt_pred     — lowest-Dice samples, CMY overlay
      viz/worst_pred_mask   — lowest-Dice samples, predicted mask

    Collapse / explosion warnings are also emitted here.
    """

    def __init__(self, n_viz: int = 8, n_best_worst: int = 4) -> None:
        self.n_viz = n_viz
        self.n_best_worst = n_best_worst
        self._reset_buffers()

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if outputs is None:
            return

        preds = outputs["preds"]  # (B, H, W) float on device
        gt = outputs["gt"]  # (B, H, W) long on device
        B = preds.shape[0]

        if batch_idx == 0:
            self._check_mask_integrity(batch["labels"])

        gt_bin = batch["qry_touch_bin"].to(preds.device)  # (B, H, W) binary float
        self._val_pred_pos_acc += preds.mean().item()
        self._val_gt_pos_acc += gt_bin.mean().item()
        self._val_n_steps += 1

        dice_scores = self._per_sample_dice(preds, gt)  # (B,)
        for i in range(B):
            dice_i = dice_scores[i].item()
            entry = {
                "pixel_values": batch["pixel_values"][i].cpu(),
                "gt_bin": gt_bin[i].cpu(),
                "pred_bin": preds[i].cpu(),
                "dice": dice_i,
            }
            self._val_best.append((dice_i, entry))
            self._val_best.sort(key=lambda x: x[0], reverse=True)
            self._val_best = self._val_best[: self.n_best_worst]

            self._val_worst.append((dice_i, entry))
            self._val_worst.sort(key=lambda x: x[0])
            self._val_worst = self._val_worst[: self.n_best_worst]

        n_buffered = len(self._val_viz_buffer)
        if n_buffered < self.n_viz:
            for i in range(min(self.n_viz - n_buffered, B)):
                self._val_viz_buffer.append(
                    {
                        "pixel_values": batch["pixel_values"][i].cpu(),
                        "gt_bin": gt_bin[i].cpu(),
                        "pred_bin": preds[i].cpu(),
                    }
                )

    def on_validation_epoch_end(self, trainer, pl_module):
        logger = trainer.logger
        if logger is not None:
            if self._val_viz_buffer:
                self._log_validation_visuals(logger)
            if self._val_best or self._val_worst:
                self._log_best_worst_visuals(logger)

        if self._val_n_steps > 0:
            mean_pred = self._val_pred_pos_acc / self._val_n_steps
            mean_gt = self._val_gt_pos_acc / self._val_n_steps
            if mean_pred < 1e-3:
                rank_zero_warn(
                    f"[prediction collapse] Mean val predicted-positive ratio is near zero "
                    f"({mean_pred:.5f}). Model may have collapsed to all-negative predictions."
                )
            if mean_gt > 0 and mean_pred > 10.0 * mean_gt:
                rank_zero_warn(
                    f"[prediction explosion] Mean val predicted-positive ratio ({mean_pred:.4f}) "
                    f"is >10× the GT ratio ({mean_gt:.4f}). Check for false-positive explosion."
                )

        self._reset_buffers()

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _denorm(img: torch.Tensor) -> torch.Tensor:
        """Undo ImageNet normalization; returns float (3, H, W) in [0, 1]."""
        mean = torch.tensor(_IMAGENET_MEAN, dtype=img.dtype)[:, None, None]
        std = torch.tensor(_IMAGENET_STD, dtype=img.dtype)[:, None, None]
        return (img * std + mean).clamp(0.0, 1.0)

    @staticmethod
    def _overlay(
        img: torch.Tensor,
        gt_mask: torch.Tensor,
        pred_mask: torch.Tensor,
        alpha: float = 0.5,
    ) -> torch.Tensor:
        """Blend masks onto a denormalized (3, H, W) image.

        Cyan  = GT-only   (0, 1, 1)
        Yellow = pred-only (1, 1, 0)
        Magenta = overlap  (1, 0, 1)
        These three are the CMY secondaries, evenly spaced at 120° on the color wheel.
        """
        out = img.clone()
        gt = gt_mask.bool()
        pred = pred_mask.bool()
        gt_only = gt & ~pred
        pred_only = pred & ~gt
        overlap = gt & pred

        boost = lambda c: alpha + (1.0 - alpha) * c  # noqa: E731
        dim = lambda c: (1.0 - alpha) * c  # noqa: E731

        # R: 0 for Cyan (gt_only) → dim; 1 for Yellow+Magenta (pred) → boost
        out[0] = torch.where(gt_only, dim(img[0]), img[0])
        out[0] = torch.where(pred_only | overlap, boost(img[0]), out[0])
        # G: 1 for Cyan+Yellow (gt_only|pred_only) → boost; 0 for Magenta (overlap) → dim
        out[1] = torch.where(gt_only | pred_only, boost(img[1]), img[1])
        out[1] = torch.where(overlap, dim(img[1]), out[1])
        # B: 1 for Cyan+Magenta (gt_only|overlap) → boost; 0 for Yellow (pred_only) → dim
        out[2] = torch.where(gt_only | overlap, boost(img[2]), img[2])
        out[2] = torch.where(pred_only, dim(img[2]), out[2])

        return out.clamp(0.0, 1.0)

    def _make_pred_image(self, entry: dict) -> np.ndarray:
        """Render predicted mask (white) with GT touch drawn in red on top."""
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

    def _make_overlay(self, entry: dict) -> np.ndarray:
        """CMY overlay: Cyan=GT-only, Yellow=Pred-only, Magenta=overlap."""
        img = self._denorm(entry["pixel_values"])
        gt = entry["gt_bin"].bool()
        pred = entry["pred_bin"]
        return (self._overlay(img, gt, pred).permute(1, 2, 0).numpy() * 255).astype(
            np.uint8
        )

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _per_sample_dice(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Compute Dice per sample: (B, H, W) binary → (B,) float."""
        B = pred.shape[0]
        p = pred.view(B, -1).float()
        t = gt.view(B, -1).float()
        intersection = (p * t).sum(dim=1)
        return (2.0 * intersection) / (p.sum(dim=1) + t.sum(dim=1) + 1e-6)

    @staticmethod
    def _check_mask_integrity(labels: torch.Tensor) -> None:
        """Warn if labels contain bilinear-interpolation artifacts.

        With nearest-neighbor resizing every mask pixel is exactly one of the
        class colours (including pure background).  Bilinear interpolation
        creates "blended" pixels at region boundaries that sit close to — but
        not exactly at — the background value.

        Strategy: compute the per-pixel mean-L1 distance from pure background.
        For NN masks this should be either ≈ 0 (background) or large (class
        colour).  A significant fraction of pixels in the intermediate range
        (0.05 – 0.5) indicates blending.
        """
        bg = torch.tensor(_BG_NORM, dtype=labels.dtype, device=labels.device)
        dist_from_bg = (labels - bg[:, None, None]).abs().mean(dim=1)  # (B, H, W)
        ambiguous = ((dist_from_bg > 0.05) & (dist_from_bg < 0.5)).float().mean()
        if ambiguous > 0.01:
            rank_zero_warn(
                f"[mask integrity] {ambiguous:.2%} of label pixels have a distance from "
                f"background in the intermediate range (0.05–0.5), suggesting bilinear "
                "interpolation was used when resizing masks instead of nearest-neighbor."
            )

    # ------------------------------------------------------------------
    # WandB logging
    # ------------------------------------------------------------------

    def _log_validation_visuals(self, logger) -> None:
        overlays, preds = [], []
        for idx, entry in enumerate(self._val_viz_buffer):
            caption = f"sample_{idx}"
            overlays.append(wandb.Image(self._make_overlay(entry), caption=caption))
            preds.append(wandb.Image(self._make_pred_image(entry), caption=caption))
        logger.experiment.log(
            {
                "viz/gt_pred_compare": overlays,
                "viz/pred_mask": preds,
            }
        )

    def _log_best_worst_visuals(self, logger) -> None:
        def _build(ranked: list[tuple[float, dict]], tag: str) -> tuple[list, list]:
            overlays, preds = [], []
            for rank, (dice, entry) in enumerate(ranked):
                caption = f"{tag}_{rank} (dice={dice:.3f})"
                overlays.append(wandb.Image(self._make_overlay(entry), caption=caption))
                preds.append(wandb.Image(self._make_pred_image(entry), caption=caption))
            return overlays, preds

        best_overlays, best_preds = _build(self._val_best, "best")
        worst_overlays, worst_preds = _build(self._val_worst, "worst")
        logger.experiment.log(
            {
                "viz/best_gt_pred": best_overlays,
                "viz/best_pred_mask": best_preds,
                "viz/worst_gt_pred": worst_overlays,
                "viz/worst_pred_mask": worst_preds,
            }
        )

    # ------------------------------------------------------------------
    # Internal state
    # ------------------------------------------------------------------

    def _reset_buffers(self) -> None:
        self._val_viz_buffer: list[dict] = []
        self._val_best: list[tuple[float, dict]] = []
        self._val_worst: list[tuple[float, dict]] = []
        self._val_pred_pos_acc: float = 0.0
        self._val_gt_pos_acc: float = 0.0
        self._val_n_steps: int = 0
