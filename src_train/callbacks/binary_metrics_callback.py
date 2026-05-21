"""Callback that computes pixel-level binary classification metrics every validation epoch.

Global metrics (accuracy, precision, recall, F1) are logged to wandb under
  binary/train/*  and  binary/val/*

Per-class metrics are written only to local JSON files and synced to wandb files
(never to wandb charts) so the per-class table can be post-processed freely.

Files saved per run:
  binary_performance_metrics/final.json  — updated every validation epoch
  binary_performance_metrics/best.json   — saved when val F1 improves

The callback depends on:
  • training_step returning {"loss": ..., "preds": cpu_tensor, "gt": cpu_tensor}
  • validation_step returning {"preds": tensor, "gt": tensor}
  • batch containing "category": str (added by TouchPairDataset)

Train predictions are computed under training-mode masking (random 75% mask),
so they are noisier than val predictions which use the full target-half mask.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch
import wandb
from lightning import Callback, LightningModule, Trainer


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _safe_div(n: float, d: float) -> float:
    return n / d if d > 0 else float("nan")


def _finalize(v: float) -> float | None:
    return round(v, 6) if v == v else None  # NaN check: NaN != NaN


def _aggregate(counts: dict) -> dict:
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    n = tp + fp + fn + tn
    precision = _safe_div(tp, tp + fp)
    recall    = _safe_div(tp, tp + fn)
    f1        = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else float("nan")
    accuracy  = _safe_div(tp + tn, n)
    return {
        "n": n,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy":  _finalize(accuracy),
        "precision": _finalize(precision),
        "recall":    _finalize(recall),
        "f1":        _finalize(f1),
    }


def _update(
    stats: dict,
    preds: torch.Tensor,   # (B, H, W) float on CPU
    gt: torch.Tensor,      # (B, H, W) long on CPU
    categories: list[str],
) -> None:
    B = preds.shape[0]
    for i in range(B):
        cat = categories[i] if i < len(categories) else "(unknown)"
        p = preds[i].bool()
        g = gt[i].bool()
        tp = int((p & g).sum())
        fp = int((p & ~g).sum())
        fn = int((~p & g).sum())
        tn = int((~p & ~g).sum())
        for key in (cat, "__all__"):
            s = stats[key]
            s["tp"] += tp; s["fp"] += fp
            s["fn"] += fn; s["tn"] += tn


def _new_stats():
    return defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

class BinaryMetricsCallback(Callback):
    """Accumulates binary preds/GT each epoch and writes JSON metric files.

    Parameters
    ----------
    monitor:
        Payload key (e.g. "f1") used to decide when to overwrite best.json.
        Always compared against the val split's global metrics.
    mode:
        "max" (default) or "min" for the monitor metric.
    """

    _FOLDER = "binary_performance_metrics"

    def __init__(self, monitor: str = "f1", mode: str = "max") -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")
        self.monitor = monitor
        self.mode = mode
        self._best_score: float = float("inf") if mode == "min" else float("-inf")
        self._train_stats = _new_stats()
        self._val_stats   = _new_stats()

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx
    ) -> None:
        if not isinstance(outputs, dict):
            return
        preds = outputs.get("preds")
        gt    = outputs.get("gt")
        if preds is None or gt is None:
            return
        _update(self._train_stats, preds.detach().cpu().float(), gt.detach().cpu(), list(batch.get("category", [])))

    def on_validation_batch_end(
        self, trainer: Trainer, pl_module: LightningModule,
        outputs, batch, batch_idx, dataloader_idx: int = 0,
    ) -> None:
        if not isinstance(outputs, dict):
            return
        preds = outputs.get("preds")
        gt    = outputs.get("gt")
        if preds is None or gt is None:
            return
        _update(
            self._val_stats,
            preds.detach().cpu().float(), gt.detach().cpu(),
            list(batch.get("category", [])),
        )

    # ------------------------------------------------------------------
    # Epoch-end: compute, log, save
    # ------------------------------------------------------------------

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if trainer.sanity_checking:
            return

        payload = self._build_payload(trainer.current_epoch, trainer.global_step)
        self._log_global_to_wandb(trainer, payload)
        self._save(payload, "final", trainer)

        val_score = (payload["val"]["global"].get(self.monitor) or 0.0)
        if self._is_better(val_score):
            self._best_score = val_score
            self._save(payload, "best", trainer)
            print(
                f"  [BinaryMetrics] New best val/{self.monitor}="
                f"{val_score:.4f} — saved best metrics."
            )

        # Train stats reset after each validation (covers the intervening training steps).
        # Val stats reset here since a fresh val pass will run next epoch.
        self._train_stats = _new_stats()
        self._val_stats   = _new_stats()

    # ------------------------------------------------------------------
    # Build payload
    # ------------------------------------------------------------------

    def _build_payload(self, epoch: int, step: int) -> dict:
        def _split_dict(stats: dict) -> dict:
            global_counts = stats.get("__all__", {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
            per_class = {
                cls: _aggregate(s)
                for cls, s in sorted(stats.items())
                if cls != "__all__"
            }
            return {"global": _aggregate(global_counts), "per_class": per_class}

        return {
            "epoch": epoch,
            "step": step,
            "note": (
                "train preds use random 75%-masked forward (training mode) "
                "and are noisier than val preds which use the full target-half mask."
            ),
            "train": _split_dict(self._train_stats),
            "val":   _split_dict(self._val_stats),
        }

    # ------------------------------------------------------------------
    # wandb logging / file saving
    # ------------------------------------------------------------------

    def _log_global_to_wandb(self, trainer: Trainer, payload: dict) -> None:
        if wandb.run is None or trainer.logger is None:
            return
        log_dict = {}
        for split in ("train", "val"):
            g = payload[split]["global"]
            for metric in ("accuracy", "precision", "recall", "f1"):
                val = g.get(metric)
                if val is not None:
                    log_dict[f"binary/{split}/{metric}"] = val
        trainer.logger.experiment.log(log_dict)

    def _save(self, payload: dict, variant: str, trainer: Trainer) -> None:
        if wandb.run is None:
            return
        folder = Path(wandb.run.dir) / self._FOLDER
        folder.mkdir(exist_ok=True)
        out = folder / f"{variant}.json"
        out.write_text(json.dumps(payload, indent=2))
        wandb.save(str(out), base_path=str(Path(wandb.run.dir)))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_better(self, current: float) -> bool:
        return current < self._best_score if self.mode == "min" else current > self._best_score
