import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
import json
from PIL import Image
from pathlib import Path
from typing import Any

VISUAL_PROMPT_WITHOUT_AUDIO = """
You are given an image.
    
Task:
Determine whether the image contains a clear physical touch or contact event between two objects, two people, or a person and an object. 

Examples of touch/contact:
- Holding, grabbing, pushing, hugging
- Hand touching an object
- Physical interaction with surfaces or tools
- Collision or impact

Output rules:
- Return only:
    1 = physical touch/contact is present
    0 = no physical touch/contact is visible
- Do not explain your answer.
- If the contact is ambiguous or unclear, return 0.
"""

# ── W&B Setup Logic ───────────────────────────────────────────────────

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
            "wandb is not installed. Install it via 'pip install wandb' or recreate the environment."
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


# ── Data loading ────────────────────────────────────────────────────────────────

def load_dataset_from_json(json_path):
    """
    Expects a JSON file with a list of records:
      [{"image_path": "path/to/frame.jpg", "type": "touch"}, ...]
    where type is "touch" or "no_touch".
    """
    with open(json_path) as f:
        records = json.load(f)
    return Dataset.from_list(records)


class QwenTouchCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        """
        Processes images and text dynamically for the current batch only.
        """
        texts = []
        images = []
        
        for record in features:
            image = Image.open(record["image_path"]).convert("RGB")
            image.thumbnail((768, 768), Image.Resampling.LANCZOS)
            label_text = "1" if record["type"] == "touch" else "0"

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": VISUAL_PROMPT_WITHOUT_AUDIO},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": label_text}],
                },
            ]
            
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
            images.append(image)

        # Process the batch dynamically
        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        # Create labels and mask padding tokens so they don't affect loss
        labels = batch["input_ids"].clone()
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            
        batch["labels"] = labels
        return batch


# def format_sample(record, processor):
#     """
#     Converts a single record into the chat-style format Qwen2-VL expects.
#     The model is prompted to answer '1' or '0' for touch detection.
#     """
#     image = Image.open(record["image_path"]).convert("RGB")
#     MAX_SIZE = (768, 768) 
#     image.thumbnail(MAX_SIZE, Image.Resampling.LANCZOS)
#     label_text = "1" if record["type"] == "touch" else "0"

#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "image", "image": image},
#                 {"type": "text", "text": VISUAL_PROMPT_WITHOUT_AUDIO},
#             ],
#         },
#         {
#             "role": "assistant",
#             "content": [{"type": "text", "text": label_text}],
#         },
#     ]

#     # Apply the chat template to get the final text
#     text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

#     # Process image + text together
#     inputs = processor(
#         text=[text],
#         images=[image],
#         return_tensors="pt",
#         padding=True,
#     )

#     # Flatten the batch dimension added by the processor
#     inputs = {k: v.squeeze(0) for k, v in inputs.items()}
#     inputs["labels"] = inputs["input_ids"].clone()
#     return inputs


# def build_hf_dataset(json_path, processor):
#     records = load_dataset_from_json(json_path)
#     formatted = [format_sample(r, processor) for r in records]
#     return Dataset.from_list(formatted)


# ── Model loading ────────────────────────────────────────────────────────────────

def load_quantized_model(model_id):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )
    processor = Qwen2VLProcessor.from_pretrained(model_id)

    return model, processor  


# ── Fine-tuning ──────────────────────────────────────────────────────────────────

def finetuning(model, processor, train_dataset, eval_dataset, wandb_run=None):
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir="qwen2-vl-touch-detection",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        learning_rate=2e-4,
        logging_steps=10,
        eval_steps=10,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=20,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        push_to_hub=False,       # set True if you want to push to HF Hub
        report_to="wandb",        # swap for "wandb" / "trackio" if needed
        remove_unused_columns=False,  # required for multimodal datasets
        dataset_kwargs={"skip_prepare_dataset": True},  # we pre-process ourselves
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        data_collator=QwenTouchCollator(processor),
        # processing_class=processor,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)

    # Clean up wandb run at the end if it was active
    if wandb_run is not None:
        import wandb
        wandb.finish()


# ── Entry point ──────────────────────────────────────────────────────────────────

def main(args):
    model_id = "Qwen/Qwen2-VL-7B-Instruct"  # fix: corrected model name

    config = _args_config(args)
    wandb_run = _maybe_init_wandb(args, config)

    model, processor = load_quantized_model(model_id)
    
    train_dataset = load_dataset_from_json(args.train_path)
    eval_dataset  = load_dataset_from_json(args.eval_path)

    finetuning(model, processor, train_dataset, eval_dataset, wandb_run)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2-VL for touch detection")
    parser.add_argument("--train_path", type=str, required=True, help="Path to training JSON file")
    parser.add_argument("--eval_path", type=str, required=True, help="Path to evaluation JSON file")
    
    # W&B configuration arguments mapped from train.py
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="qwen25-vl-finetuning", help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity (team/user)")
    parser.add_argument("--wandb_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--wandb_group", type=str, default=None, help="W&B group name")
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=None, help="W&B tags")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"], help="W&B tracking mode")
    
    args = parser.parse_args()
    main(args)