from __future__ import annotations
import json
from pathlib import Path
from typing import Any

import torch
from transformers import BitsAndBytesConfig, AutoModelForImageTextToText, AutoProcessor, EarlyStoppingCallback
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

# Import the shared splitting logic from DINO
from dino.dataset import load_splits, filter_point_samples

# Import our new Qwen collator
from .dataset import QwenTouchCollator


def _jsonable_config(value: Any):
    """Recursively formats config dictionaries for Weights & Biases."""
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
    return config


def _maybe_init_wandb(args, config: dict[str, Any]):
    if not getattr(args, "wandb", False):
        return None
    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "wandb is not installed. Install it via 'pip install wandb'."
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


def load_quantized_model(model_id: str):
    """Loads Qwen3-VL in 4-bit precision."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    processor = AutoProcessor.from_pretrained(model_id)
    if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )
    return model, processor


def train(args):
    # 1. Setup outputs and logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = _args_config(args)
    wandb_run = _maybe_init_wandb(args, config)

    # 2. Data Loading (Cross-pollinated with DINO)
    # This natively applies max_samples=250 and handles stratified splitting!
    splits = load_splits(
        datasets=args.datasets,
        annotation_roots=args.annotation_root,
        auto_split=args.auto_split,
        auto_split_test_size=args.auto_split_test_size,
        seed=args.seed,
        max_samples=args.max_samples,
    )

    if args.task == "point":
        train_filtered, train_skipped = filter_point_samples(splits.train)
        test_filtered, test_skipped = filter_point_samples(splits.test)
        print(f"Point task: Skipped {train_skipped} train and {test_skipped} eval non-touch samples.")

        # Override the splits with the filtered data
        splits = type(splits)(train=train_filtered, test=test_filtered)

    print(f"Loaded {len(splits.train)} train samples and {len(splits.test)} eval samples.")
    
    # Update W&B config with exact dataset sizes
    config.update({
        "num_train_samples": len(splits.train),
        "num_eval_samples": len(splits.test),
    })
    if wandb_run is not None:
        wandb_run.config.update(config, allow_val_change=True)

    # 3. Model & Processor Loading
    print(f"Loading base model: {args.model_id}")
    model, processor = load_quantized_model(args.model_id)

    # 4. Parameter-Efficient Fine-Tuning (LoRA) Config
    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        r=args.lora_r,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )

    # 5. Training Arguments
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        learning_rate=args.learning_rate,
        logging_steps=10,
        eval_steps=10,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=10,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        report_to="wandb" if wandb_run else "none",
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        load_best_model_at_end=True,       # Restore best weights when finished
        metric_for_best_model="eval_loss", # Track validation loss
        greater_is_better=False,           # Lower loss is better
        save_total_limit=2,
    )

    # 6. Initialization & Training Loop
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=splits.train,
        eval_dataset=splits.test,
        peft_config=peft_config,
        data_collator=QwenTouchCollator(processor, task=args.task),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )

    print("Starting fine-tuning...")
    trainer.train()
    
    # 7. Saving & Artifact Management
    final_checkpoint_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_checkpoint_dir))
    processor.save_pretrained(str(final_checkpoint_dir))
    print(f"Saved final LoRA adapters to {final_checkpoint_dir}")

    # Mirroring DINO's robust Artifact logging to save the adapters to the cloud
    if wandb_run is not None:
        import wandb
        artifact = wandb.Artifact(
            name=f"{wandb_run.name or 'qwen'}-adapter-weights",
            type="qwen-lora-output",
        )
        artifact.add_dir(str(final_checkpoint_dir))
        wandb_run.log_artifact(artifact)
        wandb_run.finish()