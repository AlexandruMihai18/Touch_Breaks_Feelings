"""
Fine-tunes SegGPT for touch-region segmentation.

SegGPT is an in-context learner: it takes a (context_image, context_mask) pair
and predicts the corresponding mask for a new target image. Fine-tuning it on
touch-region masks teaches it to identify contact zones between two objects.

Key training details (aligned with the SegGPT / Painter papers)
---------------------------------------------------------------
- Random colour mapping: both context and query masks get the same random RGB
  colour per sample, preventing the model from learning a fixed colour→task rule.
- Block-wise masking: ~75% of target patches are masked during training
  (MaskingGenerator with BEiT-style block masking). Validation always uses
  full target-half masking (the standard inference mode).
- embedding_type="instance": touch is a spatial, instance-level relationship.
- Whole-model fine-tuning with AdamW + OneCycleLR

Data preparation
----------------
1. Extract frames:   python preprocess_frames.py --video_dir ... --output_dir frames/
2. Generate masks:   <your mask pipeline>
3. Build JSON:       python src/generate_annotations.py --frames_dir frames/ --masks_dir masks/ --output_dir annotations/
4. Train:            python train.py --data_root annotations/ --output_dir checkpoints/
"""

import argparse
import random
import sys
import traceback
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from transformers import SegGptImageProcessor

import wandb

sys.path.insert(0, str(Path(__file__).parent / "src_train"))
from src_train.callbacks import BinaryMetricsCallback, CoverageBinMetricsCallback, SegGPTCheckpointCallback, SegGPTLoggingCallback
from src_train.lit_module import SegGPTTouchModule
from src_train.touch_dataset import TouchPairDataset
from src_train.wandb_utils import load_pretrained_model

SEGGPT_ID = "BAAI/seggpt-vit-large"

_PROJECT_ROOT = Path(__file__).parent

# Registry mapping --datasets keys to annotation directories.
DATASET_REGISTRY: dict[str, Path] = {
    "epic_kitchen": _PROJECT_ROOT / "data" / "epic_kitchen"       / "annotations",
    "greatest_hits": _PROJECT_ROOT / "data" / "greatest_hits"     / "annotations",
    "manual":       _PROJECT_ROOT / "data" / "manual_annotations" / "annotations",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune SegGPT for touch-region segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASET_REGISTRY),
        default=["epic_kitchen"],
        metavar="DATASET",
        help=(
            "One or more dataset keys to train on. "
            f"Available: {', '.join(DATASET_REGISTRY)}. "
            "Samples from all selected datasets are concatenated."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--model_id", type=str, default=SEGGPT_ID)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1,
                        help="Accumulate gradients over N batches before stepping (effective batch = batch_size × N).")
    parser.add_argument("--val_every_n_steps", type=int, default=25)
    parser.add_argument(
        "--use_encoder",
        action="store_true",
        help="Unfreeze the ViT encoder; only train the decoder",
    )
    parser.add_argument(
        "--touch-variant",
        type=str,
        default=None,
        metavar="SUFFIX",
        help=(
            "Load a precomputed touch-mask variant instead of the original *_touch.png. "
            "The suffix is appended to the target_path stem: e.g. 'refined' loads "
            "*_touch_refined.png, 'dilated_r20' loads *_touch_dilated_r20.png, "
            "'refined_dilated_r20' loads *_touch_refined_dilated_r20.png. "
            "Falls back to the original mask when the variant file is not found."
        ),
    )
    parser.add_argument(
        "--offline", action="store_true", help="Run wandb in offline mode"
    )
    parser.add_argument(
        "--wandb_run_id",
        type=str,
        default=None,
        metavar="RUN_ID",
        help=(
            "wandb run ID to resume decoder weights from (e.g. 'abc123'). "
            "Use --wandb_run_variant to pick 'best' (default) or 'last'."
        ),
    )
    parser.add_argument(
        "--wandb_run_variant",
        type=str,
        default="best",
        choices=["best", "last"],
        help="Which saved variant to load when --wandb_run_id is given.",
    )
    parser.add_argument("--project_name", type=str, default="tests")
    parser.add_argument("--group_tag", type=str, default="seggpt-touch")
    parser.add_argument(
        "--filter_classes",
        nargs="+",
        default=None,
        metavar="OBJECT_NAME",
        help=(
            "Restrict training and validation to one or more object classes (EPIC Kitchen "
            "object_name / video_id key). Pass 'auto' to auto-select the class with the "
            "most contact frames (e.g. --filter_classes auto), a single name "
            "(e.g. --filter_classes knife), or multiple names "
            "(e.g. --filter_classes knife spoon)."
        ),
    )
    parser.add_argument(
        "--object-filter-file",
        default=None,
        metavar="PATH",
        help=(
            "Path to a labels file (e.g. data/epic_kitchen/object_labels_filtered.txt) "
            "whose non-comment lines list allowed object names. "
            "Merged with --filter_classes if both are given. "
            "Line format: '<count>  <name>' or just '<name>'."
        ),
    )
    return parser.parse_args()


def _load_object_filter_file(path: str) -> list[str]:
    names = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)   # split off leading count if present
        name = parts[1] if len(parts) == 2 and parts[0].isdigit() else parts[0]
        names.append(name.strip())
    return names


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_dataloaders(args, processor):
    missing = [d for d in args.datasets if not (DATASET_REGISTRY[d] / "train.json").exists()]
    if missing:
        raise FileNotFoundError(
            f"No train.json found for: {missing}. "
            "Run generate_annotations.py (or regen_paired_masks.py) first."
        )

    train_paths = [DATASET_REGISTRY[d] / "train.json" for d in args.datasets]
    val_paths   = [DATASET_REGISTRY[d] / "val.json"   for d in args.datasets]

    # Merge --object-filter-file names into filter_classes
    filter_classes = list(args.filter_classes) if args.filter_classes else []
    if args.object_filter_file:
        filter_classes.extend(_load_object_filter_file(args.object_filter_file))
    object_filter = filter_classes if filter_classes else None

    print(f"\nDatasets: {args.datasets}")
    if object_filter:
        print(f"Object filter: {len(object_filter)} classes")
    train_ds = TouchPairDataset(train_paths, processor, augment=True,  object_filter=object_filter, touch_suffix=args.touch_variant)
    val_ds   = TouchPairDataset(val_paths,   processor, augment=False, object_filter=object_filter, touch_suffix=args.touch_variant)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=4 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=4 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )
    return train_loader, val_loader


def main():
    args = parse_args()

    set_seed(args.seed)
    torch.set_float32_matmul_precision("medium")

    run = wandb.init(
        entity="touchgpt",
        project=args.project_name,
        group=args.group_tag,
        mode="offline" if args.offline else "online",
        config=vars(args),
    )

    try:
        processor = SegGptImageProcessor.from_pretrained(args.model_id)
        train_loader, val_loader = build_dataloaders(args, processor)

        n_train_batches = len(train_loader)
        val_check_interval = min(args.val_every_n_steps, n_train_batches)
        log_every_n_steps = min(10, n_train_batches)
        if val_check_interval != args.val_every_n_steps:
            print(
                f"  val_check_interval clamped {args.val_every_n_steps} → {val_check_interval} "
                f"(only {n_train_batches} train batches)"
            )

        module = SegGPTTouchModule(
            model_id=args.model_id,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            total_steps=args.max_steps,
            freeze_encoder=not args.use_encoder,
        )

        if args.wandb_run_id:
            load_pretrained_model(args.wandb_run_id, module, variant=args.wandb_run_variant)

        trainer = L.Trainer(
            max_steps=args.max_steps,
            val_check_interval=val_check_interval,
            default_root_dir=args.output_dir,
            log_every_n_steps=log_every_n_steps,
            precision="16-mixed",
            accumulate_grad_batches=args.accumulate_grad_batches,
            logger=WandbLogger(run=run, save_dir="."),
            callbacks=[
                SegGPTLoggingCallback(),
                SegGPTCheckpointCallback(monitor="val/iou", mode="max"),
                LearningRateMonitor(logging_interval="step"),
                BinaryMetricsCallback(monitor="val/iou", mode="max"),
                CoverageBinMetricsCallback(),
            ],
        )

        trainer.fit(module, train_loader, val_loader)

    except KeyboardInterrupt:
        print("\nTraining interrupted.")
        wandb.finish()
    except Exception:
        print("\nAn error occurred during training:")
        traceback.print_exc()
        wandb.finish(exit_code=1)
        raise
    else:
        wandb.finish()


if __name__ == "__main__":
    main()
