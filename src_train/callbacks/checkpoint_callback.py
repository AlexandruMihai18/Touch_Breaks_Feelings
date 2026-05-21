from lightning.pytorch import Callback, LightningModule, Trainer

from src_train.wandb_utils import save_and_log_model


class SegGPTCheckpointCallback(Callback):
    """Saves decoder weights to wandb files every validation step.

    Always saves a 'last' snapshot. Saves 'best' when the monitored metric improves.
    Weights are logged as wandb files (not artifacts) under individual_components/.
    """

    def __init__(self, monitor: str = "val/iou", mode: str = "max"):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")
        self.monitor = monitor
        self.mode = mode
        self._best_score = float("inf") if mode == "min" else float("-inf")

    def _is_better(self, current: float) -> bool:
        return current < self._best_score if self.mode == "min" else current > self._best_score

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.sanity_checking:
            return

        save_and_log_model(pl_module, variant="last")

        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return
        current = current.item() if hasattr(current, "item") else float(current)
        if self._is_better(current):
            self._best_score = current
            save_and_log_model(pl_module, variant="best")
            print(f"  New best {self.monitor}={current:.4f} — saved best decoder.")
