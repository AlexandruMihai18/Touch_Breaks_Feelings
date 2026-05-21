from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from .dataset import DinoFrameDataset, DinoSample, load_splits, split_samples
from .evaluate import compute_metrics, device_from_args, run_predictions
from .hf import auth_kwargs, explain_hf_load_error
from .model import DinoClassifier


DEFAULT_MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"


def _jsonable_config(value: Any):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable_config(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable_config(v) for k, v in value.items()}
    return str(value)


def _args_config(args) -> dict[str, Any]:
    config = {
        key: _jsonable_config(value)
        for key, value in vars(args).items()
        if key != "hf_token"
    }
    config["hf_token_provided"] = bool(getattr(args, "hf_token", None))
    return config


def _maybe_init_wandb(args, config: dict[str, Any]):
    if not getattr(args, "wandb", False):
        return None
    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "wandb is not installed. Install it or recreate the DINO environment."
        ) from exc

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        group=args.wandb_group,
        tags=args.wandb_tags,
        mode=args.wandb_mode,
        config=config,
    )


def _run_epoch(
    model: DinoClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    train = optimizer is not None
    model.head.train(train)
    total_loss = 0.0
    labels: list[int] = []
    preds: list[int] = []

    for batch in loader:
        pixel_values, batch_labels = batch[:2]
        pixel_values = pixel_values.to(device)
        batch_labels = batch_labels.to(device)
        logits = model(pixel_values)
        loss = criterion(logits, batch_labels)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * len(batch_labels)
        labels.extend(batch_labels.cpu().tolist())
        preds.extend(torch.argmax(logits.detach(), dim=1).cpu().tolist())

    return total_loss / len(loader.dataset), accuracy_score(labels, preds)


def _build_loaders(
    train_samples: list[DinoSample],
    val_samples: list[DinoSample],
    test_samples: list[DinoSample],
    processor,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(
        DinoFrameDataset(train_samples, processor),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        DinoFrameDataset(val_samples, processor),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        DinoFrameDataset(test_samples, processor),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader


def _save_loss_plot(train_losses: list[float], val_losses: list[float], path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="train")
    ax.plot(epochs, val_losses, label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("DINO classifier training")
    ax.legend()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train(args) -> dict:
    from transformers import AutoImageProcessor, AutoModel

    device = device_from_args(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _args_config(args)
    config["resolved_device"] = str(device)
    wandb_run = _maybe_init_wandb(args, config)

    splits = load_splits(
        datasets=args.datasets,
        annotation_roots=args.annotation_root,
        auto_split=args.auto_split,
        auto_split_test_size=args.auto_split_test_size,
        seed=args.seed,
        max_samples=args.max_samples,
    )
    train_samples, val_samples = split_samples(splits.train, args.val_split, args.seed)
    test_samples = splits.test

    print(
        f"Samples: train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}"
    )
    config.update(
        {
            "num_train_samples": len(train_samples),
            "num_val_samples": len(val_samples),
            "num_test_samples": len(test_samples),
        }
    )
    if wandb_run is not None:
        wandb_run.config.update(config, allow_val_change=True)

    print(f"Loading model: {args.model_id}")
    try:
        processor = AutoImageProcessor.from_pretrained(
            args.model_id,
            **auth_kwargs(args.hf_token),
        )
        encoder = AutoModel.from_pretrained(
            args.model_id,
            torch_dtype=torch.float32,
            **auth_kwargs(args.hf_token),
        )
    except OSError as exc:
        raise explain_hf_load_error(exc, args.model_id) from exc
    encoder.eval()

    model = DinoClassifier(encoder, head_type=args.head_type).to(device)
    optimizer = Adam(model.head.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    train_loader, val_loader, test_loader = _build_loaders(
        train_samples,
        val_samples,
        test_samples,
        processor,
        args.batch_size,
        args.num_workers,
    )

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    best_head_state = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, None, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/accuracy": train_acc,
                    "val/loss": val_loss,
                    "val/accuracy": val_acc,
                },
                step=epoch,
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_head_state = {k: v.cpu().clone() for k, v in model.head.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_head_state is None:
        best_head_state = {k: v.cpu().clone() for k, v in model.head.state_dict().items()}
    model.head.load_state_dict(best_head_state)

    checkpoint_path = output_dir / "best_head.pt"
    torch.save(
        {
            "head_state_dict": best_head_state,
            "model_id": args.model_id,
            "hidden_size": getattr(encoder.config, "hidden_size", None),
            "head_type": args.head_type,
            "label_map": {"no-touch": 0, "touch": 1},
        },
        checkpoint_path,
    )
    _save_loss_plot(train_losses, val_losses, output_dir / "dino_training_loss.png")
    plot_path = output_dir / "dino_training_loss.png"

    preds, labels, datasets, video_ids, frame_ids = run_predictions(model, test_loader, device)
    metrics = compute_metrics(preds, labels, datasets)
    metrics.update(
        {
            "best_val_loss": best_val_loss,
            "checkpoint": str(checkpoint_path),
            "model_id": args.model_id,
            "head_type": args.head_type,
            "hyperparameters": config,
        }
    )
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    predictions_path = output_dir / "predictions.csv"
    from .evaluate import build_prediction_rows, save_predictions_csv

    save_predictions_csv(
        build_prediction_rows(preds, labels, video_ids, frame_ids),
        predictions_path,
    )
    if wandb_run is not None:
        import wandb

        wandb_run.log(
            {
                "test/accuracy": metrics["accuracy"],
                "test/precision": metrics["precision"],
                "test/recall": metrics["recall"],
                "test/f1": metrics["f1"],
                "test/tp": metrics["tp"],
                "test/fp": metrics["fp"],
                "test/tn": metrics["tn"],
                "test/fn": metrics["fn"],
                "best_val_loss": best_val_loss,
            }
        )
        artifact = wandb.Artifact(
            name=f"{wandb_run.name or 'dino'}-outputs",
            type="dino-training-output",
        )
        for path in (checkpoint_path, metrics_path, predictions_path, plot_path):
            artifact.add_file(str(path))
        wandb_run.log_artifact(artifact)
        wandb_run.finish()

    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved predictions: {predictions_path}")
    return metrics
