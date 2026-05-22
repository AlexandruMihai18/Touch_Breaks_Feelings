import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
import json
from PIL import Image
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


# ── Data loading ────────────────────────────────────────────────────────────────

def load_dataset_from_json(json_path):
    """
    Expects a JSON file with a list of records:
      [{"image_path": "path/to/frame.jpg", "type": "touch"}, ...]
    where type is "touch" or "no_touch".
    """
    with open(json_path) as f:
        records = json.load(f)
    return records


def format_sample(record, processor):
    """
    Converts a single record into the chat-style format Qwen2-VL expects.
    The model is prompted to answer '1' or '0' for touch detection.
    """
    image = Image.open(record["image_path"]).convert("RGB")
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

    # Apply the chat template to get the final text
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    # Process image + text together
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
    )

    # Flatten the batch dimension added by the processor
    inputs = {k: v.squeeze(0) for k, v in inputs.items()}
    inputs["labels"] = inputs["input_ids"].clone()
    return inputs


def build_hf_dataset(json_path, processor):
    records = load_dataset_from_json(json_path)
    formatted = [format_sample(r, processor) for r in records]
    return Dataset.from_list(formatted)


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

def finetuning(model, processor, train_dataset, eval_dataset):
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
        processing_class=processor,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)


# ── Entry point ──────────────────────────────────────────────────────────────────

def main(train_path, eval_path):
    model_id = "Qwen/Qwen2-VL-7B-Instruct"  # fix: corrected model name

    model, processor = load_quantized_model(model_id)
    
    train_dataset = build_hf_dataset(train_path, processor)
    eval_dataset  = build_hf_dataset(eval_path,  processor)

    finetuning(model, processor, train_dataset, eval_dataset)


if __name__ == "__main__":
    train_path = "data/epic_kitchens/annotations/train.json"
    eval_path = "data/epic_kitchens/annotations/val.json"
    main(train_path, eval_path)