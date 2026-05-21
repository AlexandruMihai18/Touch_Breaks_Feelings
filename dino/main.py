from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .dataset import DATASET_REGISTRY, load_splits
from .evaluate import device_from_args, evaluate_samples, save_metrics
from .hf import auth_kwargs, explain_hf_load_error
from .model import HEAD_TYPES
from .train import DEFAULT_MODEL_ID, train


def _add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        choices=sorted(DATASET_REGISTRY),
        help="Named annotation datasets to use.",
    )
    parser.add_argument(
        "--annotation-root",
        nargs="+",
        default=None,
        type=Path,
        help="One or more custom annotation directories containing train/test/val JSON files.",
    )
    parser.add_argument(
        "--auto-split",
        action="store_true",
        help="Split a single available JSON file in memory when no explicit train/test pair exists.",
    )
    parser.add_argument("--auto-split-test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or evaluate a frozen-DINO touch/no-touch classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    _add_data_args(train_parser)
    train_parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    train_parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional Hugging Face token for gated/private model checkpoints.",
    )
    train_parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/dino"))
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--val-split", type=float, default=0.2)
    train_parser.add_argument("--patience", type=int, default=5)
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument(
        "--head-type",
        choices=HEAD_TYPES,
        default="mlp",
        help="Classifier head architecture: one linear layer or a two-layer MLP.",
    )
    train_parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    train_parser.add_argument("--wandb-project", default="dino-touch")
    train_parser.add_argument("--wandb-entity", default=None)
    train_parser.add_argument("--wandb-name", default=None)
    train_parser.add_argument("--wandb-group", default=None)
    train_parser.add_argument("--wandb-tags", nargs="*", default=None)
    train_parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="online",
    )

    eval_parser = subparsers.add_parser("evaluate")
    _add_data_args(eval_parser)
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--model-id", default=None)
    eval_parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional Hugging Face token for gated/private model checkpoints.",
    )
    eval_parser.add_argument("--batch-size", type=int, default=32)
    eval_parser.add_argument("--num-workers", type=int, default=0)
    eval_parser.add_argument("--device", default="auto")
    eval_parser.add_argument(
        "--head-type",
        choices=HEAD_TYPES,
        default=None,
        help="Override classifier head architecture. Defaults to the value saved in the checkpoint.",
    )
    eval_parser.add_argument("--metrics-out", type=Path, default=Path("checkpoints/dino/eval_metrics.json"))
    eval_parser.add_argument(
        "--predictions-out",
        type=Path,
        default=Path("checkpoints/dino/eval_predictions.csv"),
        help="CSV path for per-sample predictions.",
    )

    return parser.parse_args()


def evaluate_command(args: argparse.Namespace) -> dict:
    from transformers import AutoImageProcessor, AutoModel

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_id = args.model_id or (
        checkpoint.get("model_id") if isinstance(checkpoint, dict) else None
    ) or DEFAULT_MODEL_ID

    splits = load_splits(
        datasets=args.datasets,
        annotation_roots=args.annotation_root,
        auto_split=args.auto_split,
        auto_split_test_size=args.auto_split_test_size,
        seed=args.seed,
        max_samples=args.max_samples,
    )
    device = device_from_args(args.device)
    try:
        processor = AutoImageProcessor.from_pretrained(
            model_id,
            **auth_kwargs(args.hf_token),
        )
        encoder = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            **auth_kwargs(args.hf_token),
        )
    except OSError as exc:
        raise explain_hf_load_error(exc, model_id) from exc
    encoder.eval()

    metrics = evaluate_samples(
        samples=splits.test,
        processor=processor,
        encoder=encoder,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        head_type=args.head_type,
        predictions_out=args.predictions_out,
    )
    metrics["model_id"] = model_id
    save_metrics(metrics, args.metrics_out)
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics: {args.metrics_out}")
    print(f"Saved predictions: {args.predictions_out}")
    return metrics


def main() -> None:
    args = parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate_command(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
