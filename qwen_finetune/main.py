from __future__ import annotations
import argparse
from pathlib import Path

# Directly import the dataset registry from the DINO codebase
from dino.dataset import DATASET_REGISTRY

def _add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--datasets", nargs="+", default=None, choices=sorted(DATASET_REGISTRY), help="Named annotation datasets to use.")
    parser.add_argument("--annotation-root", nargs="+", default=None, type=Path, help="Custom annotation directories containing train/test/val JSON files.")
    parser.add_argument("--auto-split", action="store_true", help="Split single JSON in memory.")
    parser.add_argument("--auto-split-test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=250, help="Subset size for rapid testing.")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-VL for touch detection.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- TRAIN SUBPARSER ---
    train_parser = subparsers.add_parser("train")
    _add_data_args(train_parser)
    train_parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    train_parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/qwen_touch"))
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--grad-accum-steps", type=int, default=8)
    train_parser.add_argument("--learning-rate", type=float, default=2e-4)
    train_parser.add_argument("--lora-r", type=int, default=8)
    train_parser.add_argument("--lora-alpha", type=int, default=16)
    train_parser.add_argument("--patience", type=int, default=3, help="Early stopping patience (epochs).")
    train_parser.add_argument("--task", choices=["binary", "point"], default="binary", help="Task to fine-tune on.")

    # W&B arguments identical to DINO
    train_parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    train_parser.add_argument("--wandb-project", default="qwen-touch-finetuning")
    train_parser.add_argument("--wandb-entity", default=None)
    train_parser.add_argument("--wandb-name", default=None)
    train_parser.add_argument("--wandb-group", default=None)
    train_parser.add_argument("--wandb-tags", nargs="*", default=None)
    train_parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")

    # --- EVALUATE SUBPARSER (For Phase 4) ---
    eval_parser = subparsers.add_parser("evaluate")
    _add_data_args(eval_parser)
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--task", choices=["binary", "point"], default="binary", help="Task to evaluate.")

    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if args.command == "train":
        from .train import train 
        train(args)
    elif args.command == "evaluate":
        from .evaluate import evaluate_command # <-- ADD THIS
        evaluate_command(args) 
    else:
        raise ValueError(f"Unknown command: {args.command}")

if __name__ == "__main__":
    main()