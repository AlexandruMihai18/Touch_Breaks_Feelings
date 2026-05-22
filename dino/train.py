from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import numpy as np

from .dataset import DinoFrameDataset, DinoSample, filter_point_samples, load_splits, split_samples
from .evaluate import (
    build_prediction_rows,
    compute_metrics,
    compute_point_metrics,
    dataset_output_name,
    device_from_args,
    prediction_csv_name,
    run_predictions,
    save_predictions_csv,
)
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
    task: str = "binary",
) -> tuple[float, float]:
    train = optimizer is not None
    model.head.train(train)
    total_loss = 0.0
    labels: list = []
    preds: list = []

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
        if task == "point":
            labels.extend(batch_labels.cpu().tolist())
            preds.extend(logits.detach().cpu().tolist())
        else:
            labels.extend(batch_labels.cpu().tolist())
            preds.extend(torch.argmax(logits.detach(), dim=1).cpu().tolist())

    if task == "point":
        pred_arr = np.asarray(preds, dtype=np.float64)
        label_arr = np.asarray(labels, dtype=np.float64)
        metric = float(np.linalg.norm(pred_arr - label_arr, axis=1).mean())
    else:
        metric = accuracy_score(labels, preds)
    return total_loss / len(loader.dataset), metric


def _build_loaders(
    train_samples: list[DinoSample],
    val_samples: list[DinoSample],
    test_samples: list[DinoSample],
    processor,
    batch_size: int,
    num_workers: int,
    task: str,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(
        DinoFrameDataset(train_samples, processor, task=task, seed=seed),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        DinoFrameDataset(val_samples, processor, task=task, seed=seed),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        DinoFrameDataset(test_samples, processor, task=task, seed=seed),
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

    if args.head_type == "mlp" and args.mlp_layers < 2:
        raise ValueError("--mlp-layers must be >= 2 when --head-type mlp")

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
    point_skip_counts = None
    if args.task == "point":
        train_filtered, train_skipped = filter_point_samples(splits.train)
        test_filtered, test_skipped = filter_point_samples(splits.test)
        point_skip_counts = {
            "train_source_skipped": train_skipped,
            "test_source_skipped": test_skipped,
        }
        splits = type(splits)(train=train_filtered, test=test_filtered)
        if len(splits.train) < 2:
            raise ValueError("Point task needs at least two valid touch samples in the train split")
        if not splits.test:
            raise ValueError("Point task needs at least one valid touch sample in the test split")

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
    if point_skip_counts is not None:
        config.update(point_skip_counts)
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

    model = DinoClassifier(
        encoder,
        head_type=args.head_type,
        mlp_layers=args.mlp_layers,
        num_classes=2,
    ).to(device)
    optimizer = Adam(model.head.parameters(), lr=args.lr)
    criterion = nn.SmoothL1Loss() if args.task == "point" else nn.CrossEntropyLoss()

    train_loader, val_loader, test_loader = _build_loaders(
        train_samples,
        val_samples,
        test_samples,
        processor,
        args.batch_size,
        args.num_workers,
        args.task,
        args.seed,
    )

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    best_head_state = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metric = _run_epoch(
            model, train_loader, criterion, optimizer, device, task=args.task
        )
        val_loss, val_metric = _run_epoch(model, val_loader, criterion, None, device, task=args.task)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        metric_name = "mean_error" if args.task == "point" else "acc"
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_{metric_name}={train_metric:.4f} "
            f"val_loss={val_loss:.4f} val_{metric_name}={val_metric:.4f}"
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    f"train/{metric_name}": train_metric,
                    "val/loss": val_loss,
                    f"val/{metric_name}": val_metric,
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
            "mlp_layers": args.mlp_layers,
            "task": args.task,
            "label_map": {"no-touch": 0, "touch": 1},
        },
        checkpoint_path,
    )
    _save_loss_plot(train_losses, val_losses, output_dir / "dino_training_loss.png")
    plot_path = output_dir / "dino_training_loss.png"

    preds, labels, datasets, video_ids, frame_ids, frame_paths, widths, heights = run_predictions(
        model, test_loader, device, task=args.task
    )
    if args.task == "point":
        metrics = compute_point_metrics(preds, labels, datasets, widths, heights)
    else:
        metrics = compute_metrics(preds, labels, datasets)
    dataset_name = dataset_output_name(datasets)
    predictions_path = output_dir / prediction_csv_name(
        dataset_name,
        args.task,
        args.head_type,
        args.mlp_layers,
    )
    metrics.update(
        {
            "best_val_loss": best_val_loss,
            "checkpoint": str(checkpoint_path),
            "model_id": args.model_id,
            "head_type": args.head_type,
            "mlp_layers": args.mlp_layers,
            "task": args.task,
            "predictions_csv": str(predictions_path),
            "hyperparameters": config,
        }
    )
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    save_predictions_csv(
        build_prediction_rows(
            preds,
            labels,
            video_ids,
            frame_ids,
            frame_paths,
            widths,
            heights,
            task=args.task,
        ),
        predictions_path,
    )
    if wandb_run is not None:
        import wandb

        wandb_metrics = {"best_val_loss": best_val_loss}
        if args.task == "point":
            wandb_metrics.update(
                {
                    "test/mean_euclidean_error": metrics["mean_euclidean_error"],
                    "test/median_euclidean_error": metrics["median_euclidean_error"],
                    "test/mean_pixel_euclidean_error": metrics["mean_pixel_euclidean_error"],
                    "test/median_pixel_euclidean_error": metrics["median_pixel_euclidean_error"],
                }
            )
        else:
            wandb_metrics.update(
                {
                    "test/accuracy": metrics["accuracy"],
                    "test/precision": metrics["precision"],
                    "test/recall": metrics["recall"],
                    "test/f1": metrics["f1"],
                    "test/tp": metrics["tp"],
                    "test/fp": metrics["fp"],
                    "test/tn": metrics["tn"],
                    "test/fn": metrics["fn"],
                }
            )
        wandb_run.log(wandb_metrics)
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
