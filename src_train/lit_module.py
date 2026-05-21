from pathlib import Path

import lightning as L
import numpy as np
import torch
from torchmetrics.classification import (BinaryAveragePrecision, BinaryF1Score,
                                         BinaryJaccardIndex, BinaryPrecision,
                                         BinaryRecall)
from transformers import SegGptForImageSegmentation, SegGptImageProcessor

from src_train.masking_generator import MaskingGenerator

SEGGPT_ID = "BAAI/seggpt-vit-large"

# SegGPT with patch_size=16 on 448×448:
#   stitched height = 2×448 = 896 → 56 patch rows
#   stitched width  = 448        → 28 patch cols
#   total patches = 1568; target half = 784 (28×28 grid)
_PATCH_GRID = (28, 28)  # target-half patch grid
_N_MASK = 588  # 75 % of 784 target patches
_N_TOTAL = (_PATCH_GRID[0] * 2) * _PATCH_GRID[1]  # 1568

# Black (0,0,0) normalised with ImageNet stats — used as the background reference
# point for nearest-colour touch decoding in _binarise.
_BG_NORM = (-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225)  # ≈ (-2.12, -2.04, -1.80)


# =========================
# Lightning Module
# =========================
class SegGPTTouchModule(L.LightningModule):
    """SegGPT fine-tuned for touch-region segmentation."""

    _SAVE_COMPONENTS = ["decoder"]

    def __init__(
        self,
        model_id: str = SEGGPT_ID,
        lr: float = 1e-5,
        weight_decay: float = 0.05,
        warmup_steps: int = 100,
        total_steps: int = 5000,
        freeze_encoder: bool = False,
        w_fn: float = 5.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = SegGptForImageSegmentation.from_pretrained(model_id)
        self.processor = SegGptImageProcessor.from_pretrained(model_id)

        if freeze_encoder:
            for param in self.model.model.parameters():
                param.requires_grad = False

        # Block-wise masking generator for the target-half patch grid (train only)
        self.masking_generator = MaskingGenerator(
            input_size=_PATCH_GRID,
            num_masking_patches=_N_MASK,
            min_num_patches=16,
            max_num_patches=100,
        )

        self.val_iou = BinaryJaccardIndex()
        self.val_f1 = BinaryF1Score()
        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()
        # Accumulates soft scores across the epoch on CPU to avoid GPU OOM
        # with large val sets (memory scales with val_size × H × W).
        self.val_pr_auc = BinaryAveragePrecision(compute_on_cpu=True)

    def _train_masked_pos(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Block-wise 75% masking of both context and target patches independently.

        Both halves use the same three-class mask format (hand+object+touch) so
        masking context patches is safe — there is no format mismatch.  The model
        gets touch supervision from whichever half has touch pixels, doubling the
        learning signal on frames where both context and query contain touch.
        At inference only the target half is masked — see _val_masked_pos.
        """
        context_flat = self.masking_generator().flatten().astype(bool)  # (784,)
        target_flat = self.masking_generator().flatten().astype(bool)    # (784,)
        pos = (
            torch.from_numpy(np.concatenate([context_flat, target_flat]))  # (1568,)
            .unsqueeze(0)  # (1, 1568)
            .expand(batch_size, -1)  # (B, 1568)
            .to(device)
        )
        return pos

    def _val_masked_pos(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Mask the entire target half — standard inference / evaluation mode."""
        half = _N_TOTAL // 2
        # Creates a (B, 1568) bool tensor where:
        # the FIRST 784 context positions are False, and
        # the LAST 784 target positions are True,
        # indicating they should be masked.
        pos = (
            torch.cat(
                [
                    torch.zeros(half, dtype=torch.bool),
                    torch.ones(half, dtype=torch.bool),
                ]
            )
            .unsqueeze(0)
            .expand(batch_size, -1)
            .to(device)
        )
        return pos

    @staticmethod
    def _masked_pos_to_pixel_mask(
        bool_masked_pos: torch.Tensor, patch_size: int, height: int, width: int
    ) -> torch.Tensor:
        """Expand (B, N_patches) bool mask → (B, 3, height, width) float mask."""
        B = bool_masked_pos.shape[0]
        n_h, n_w = height // patch_size, width // patch_size
        flat = (
            bool_masked_pos[:, :, None]
            .repeat(1, 1, patch_size * patch_size * 3)
            .float()
        )
        flat = flat.reshape(B, n_h, n_w, patch_size, patch_size, 3)
        flat = flat.permute(0, 5, 1, 3, 2, 4)
        return flat.reshape(B, 3, height, width)

    def _focal_touch_loss(
        self,
        pred_masks: torch.Tensor,
        prompt_masks: torch.Tensor,
        labels: torch.Tensor,
        ctx_touch_bin: torch.Tensor,
        qry_touch_bin: torch.Tensor,
        bool_masked_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Smooth-L1 loss over masked patches with touch-pixel upweighting.

        Both halves share the same three-class format so masking context patches
        is safe.  Touch pixels are upweighted by w_fn in both halves symmetrically,
        giving the model a stronger gradient signal wherever touch occurs regardless
        of which frame is context and which is query.
        """
        patch_size: int = self.model.config.patch_size
        beta: float = self.model.config.beta

        ground_truth = torch.cat((prompt_masks, labels), dim=2)  # (B, 3, 2H, W)

        # Patch-level bool mask → pixel-level float mask (B, 3, 2H, W)
        spatial_mask = self._masked_pos_to_pixel_mask(
            bool_masked_pos, patch_size, ground_truth.shape[2], ground_truth.shape[3]
        )

        with torch.no_grad():
            # Touch upweighting applied symmetrically to both halves
            ctx_weights = torch.ones(
                ctx_touch_bin.shape, dtype=torch.float32, device=labels.device
            )
            ctx_weights[ctx_touch_bin.bool()] = self.hparams.w_fn

            qry_weights = torch.ones(
                qry_touch_bin.shape, dtype=torch.float32, device=labels.device
            )
            qry_weights[qry_touch_bin.bool()] = self.hparams.w_fn

            pixel_weights = torch.cat(
                (
                    ctx_weights.unsqueeze(1).expand(-1, 3, -1, -1),
                    qry_weights.unsqueeze(1).expand(-1, 3, -1, -1),
                ),
                dim=2,
            )  # (B, 3, 2H, W)

        loss = torch.nn.functional.smooth_l1_loss(
            pred_masks, ground_truth, reduction="none", beta=beta
        )
        return (loss * spatial_mask * pixel_weights).sum() / spatial_mask.sum().clamp(min=1)

    def training_step(self, batch, batch_idx):
        B = batch["pixel_values"].shape[0]
        device = batch["pixel_values"].device
        bool_masked_pos = self._train_masked_pos(B, device)

        outputs = self.model(
            pixel_values=batch["pixel_values"],
            prompt_pixel_values=batch["prompt_pixel_values"],
            prompt_masks=batch["prompt_masks"],
            bool_masked_pos=bool_masked_pos,
            embedding_type="instance",
        )
        loss = self._focal_touch_loss(
            outputs.pred_masks, batch["prompt_masks"], batch["labels"],
            batch["ctx_touch_bin"], batch["qry_touch_bin"], bool_masked_pos,
        )
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite training loss at step {self.global_step}: {loss.item()}")
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("train/lr", lr, on_step=True, on_epoch=False)

        with torch.no_grad():
            preds, gt = self._binarise(outputs, batch)
        return {"loss": loss, "preds": preds.cpu(), "gt": gt.cpu()}

    def validation_step(self, batch, batch_idx):
        B = batch["pixel_values"].shape[0]
        device = batch["pixel_values"].device
        bool_masked_pos = self._val_masked_pos(B, device)

        outputs = self.model(
            pixel_values=batch["pixel_values"],
            prompt_pixel_values=batch["prompt_pixel_values"],
            prompt_masks=batch["prompt_masks"],
            bool_masked_pos=bool_masked_pos,
            embedding_type="instance",
        )
        loss = self._focal_touch_loss(
            outputs.pred_masks, batch["prompt_masks"], batch["labels"],
            batch["ctx_touch_bin"], batch["qry_touch_bin"], bool_masked_pos,
        )
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite validation loss at step {self.global_step}: {loss.item()}")
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)

        preds, gt = self._binarise(outputs, batch)
        self.val_iou.update(preds, gt)
        self.val_f1.update(preds, gt)
        self.val_precision.update(preds, gt)
        self.val_recall.update(preds, gt)

        # Soft scores for PR AUC: distance from predicted colour to touch colour.
        # Smaller distance → higher probability of being touch.
        pred = outputs.pred_masks
        H = pred.shape[2] // 2
        pred_target = pred[:, :, H:, :]  # (B, 3, H, W)
        touch_color = batch["class_colors_norm"][:, 2, :].to(device)  # (B, 3) = color_C
        dist_to_touch = (pred_target - touch_color[:, :, None, None]).abs().mean(dim=1)
        pred_prob = (1.0 - dist_to_touch / 4.5).clamp(0.0, 1.0)
        self.val_pr_auc.update(pred_prob.flatten(), gt.flatten())

        self.log("val/pred_pos_ratio", preds.mean(), on_step=False, on_epoch=True)
        self.log("val/gt_pos_ratio", gt.float().mean(), on_step=False, on_epoch=True)

        return {"preds": preds, "gt": gt}

    def on_before_optimizer_step(self, optimizer):
        grads = [
            p.grad.norm()
            for p in self.model.parameters()
            if p.requires_grad and p.grad is not None
        ]
        if grads:
            norm = torch.stack(grads).norm()
            self.log("train/decoder_grad_norm", norm, on_step=True, on_epoch=False)

    def on_validation_epoch_end(self):
        self.log("val/iou", self.val_iou.compute(), prog_bar=True)
        self.log("val/f1", self.val_f1.compute(), prog_bar=True)
        self.log("val/precision", self.val_precision.compute())
        self.log("val/recall", self.val_recall.compute())
        self.log("val/pr_auc", self.val_pr_auc.compute())
        self.val_iou.reset()
        self.val_f1.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_pr_auc.reset()

    @staticmethod
    def _binarise(outputs, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract (B, H, W) binary touch predictions and ground-truth.

        GT is taken directly from the binary qry_touch_bin mask.

        Predictions use nearest-colour classification: each predicted pixel is
        assigned to whichever of {background, hand, object, touch} its normalised
        RGB value is closest to (mean L1).  A pixel is touch if it is closest to
        color_C (class index 3 in the all_colors tensor below).
        """
        pred = outputs.pred_masks  # (B, 3, 2H, W)
        H = pred.shape[2] // 2
        pred_target = pred[:, :, H:, :]  # (B, 3, H, W)
        B = pred_target.shape[0]
        device = pred_target.device

        gt_bin = batch["qry_touch_bin"].to(device).long()  # (B, H, W)

        # Stack candidate colours: [bg, hand, obj, touch] → (B, 4, 3)
        bg = torch.tensor(_BG_NORM, device=device, dtype=pred_target.dtype)
        class_colors = batch["class_colors_norm"].to(device)  # (B, 3, 3)
        all_colors = torch.cat(
            [bg[None, None, :].expand(B, 1, 3), class_colors], dim=1
        )  # (B, 4, 3)

        # (B, H, W, 4) mean-L1 distance from each pixel to each candidate colour
        pred_hwc = pred_target.permute(0, 2, 3, 1)  # (B, H, W, 3)
        dists = (pred_hwc.unsqueeze(3) - all_colors[:, None, None, :, :]).abs().mean(dim=-1)

        nearest = dists.argmin(dim=-1)  # (B, H, W): 0=bg, 1=hand, 2=obj, 3=touch
        pred_bin = (nearest == 3).float()
        return pred_bin, gt_bin

    def save_individual_components(self, save_path, variant: str = "last") -> None:
        """Save decoder state dict to {save_path}/decoder_{variant}.pth."""
        path = Path(save_path) / f"decoder_{variant}.pth"
        torch.save(self.model.decoder.state_dict(), path)

    def load_individual_components(self, load_path) -> None:
        """Load decoder state dict from {load_path}/decoder.pth."""
        path = Path(load_path) / "decoder.pth"
        if not path.exists():
            print(f"Warning: decoder.pth not found at {load_path}, skipping load.")
            return
        state = torch.load(path, map_location="cpu")
        self.model.decoder.load_state_dict(state)
        print(f"Loaded decoder weights from {path}.")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,
            total_steps=self.hparams.total_steps,
            pct_start=self.hparams.warmup_steps / self.hparams.total_steps,
            anneal_strategy="cos",
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
