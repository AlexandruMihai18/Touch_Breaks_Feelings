from __future__ import annotations
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
import re

# 1. Import DINO's evaluation utilities directly
from dino.dataset import load_splits, filter_point_samples
from dino.evaluate import (
    compute_metrics,
    compute_point_metrics,
    build_prediction_rows,
    save_predictions_csv,
    dataset_output_name,
)
from .dataset import VISUAL_PROMPT_WITHOUT_AUDIO, VISUAL_PROMPT_POINT_OF_TOUCH, QwenTouchCollator

def evaluate_command(args):
    # 2. Load the exact same test split
    splits = load_splits(
        datasets=args.datasets,
        annotation_roots=args.annotation_root,
        auto_split=args.auto_split,
        auto_split_test_size=args.auto_split_test_size,
        seed=args.seed,
        max_samples=args.max_samples
    )
    test_samples = splits.test
    
    if getattr(args, "task", "binary") == "point":
        test_samples, skipped = filter_point_samples(test_samples)
        print(f"Point evaluation: skipped {skipped} non-touch samples.")

    print(f"Evaluating on {len(test_samples)} samples...")

    # 3. Load Base Model + LoRA Adapters
    base_model_id = getattr(args, "model_id", "Qwen/Qwen3-VL-8B-Instruct")
    print(f"Loading Base Model: {base_model_id}")
    
    # We load in bfloat16 to fit on a standard cluster GPU during inference
    base_model = AutoModelForImageTextToText.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    print(f"Loading LoRA Adapters from: {args.checkpoint}")
    model = PeftModel.from_pretrained(base_model, args.checkpoint)
    model.eval()

    processor = AutoProcessor.from_pretrained(args.checkpoint)

    # 4. Tracking lists (matching DINO's format perfectly)
    preds, labels, datasets = [], [], []
    video_ids, frame_ids, frame_paths = [], [], []
    widths, heights = [], []

    # 5. Inference Loop
    # We process samples individually during eval. It avoids left-padding bugs 
    # common in autoregressive VLM generation and is highly robust.
    for sample in tqdm(test_samples, desc="Inference"):
        image = Image.open(sample.image_path).convert("RGB")
        widths.append(float(image.width))
        heights.append(float(image.height))

        # Resize for VRAM limits
        image.thumbnail((768, 768), Image.Resampling.LANCZOS)
        prompt = VISUAL_PROMPT_POINT_OF_TOUCH if args.task == "point" else VISUAL_PROMPT_WITHOUT_AUDIO

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=15, do_sample=False)
            # Trim the prompt tokens so we only get the model's new answer
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
        
        if args.task == "point":
            # Extract floats from format (0.xxx, 0.yyy)
            match = re.search(r'\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)', output_text)
            if match:
                pred_x, pred_y = float(match.group(1)), float(match.group(2))
            else:
                pred_x, pred_y = 0.0, 0.0
            
            # Ground truth calculation (reuse Collator's logic)
            collator = QwenTouchCollator(processor, task="point")
            gt_str = collator._extract_normalized_point(sample, int(widths[-1]), int(heights[-1]))
            gt_match = re.search(r'\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)', gt_str)
            gt_x, gt_y = float(gt_match.group(1)), float(gt_match.group(2)) if gt_match else (0.0, 0.0)

            preds.append([pred_x, pred_y])
            labels.append([gt_x, gt_y])

        else:
            preds.append(1 if "1" in output_text else 0)
            labels.append(sample.label)

        datasets.append(sample.dataset)
        video_ids.append(sample.video_id)
        frame_ids.append(sample.frame_id)
        frame_paths.append(sample.image_path)

    # 6. Generate identical DINO metrics and files
    if args.task == "point":
        metrics = compute_point_metrics(preds, labels, datasets, widths, heights)
    else:
        metrics = compute_metrics(preds, labels, datasets)

    dataset_name = dataset_output_name(datasets)
    predictions_out = args.checkpoint / f"{dataset_name}_ft_qwen_{args.task}_prediction.csv"
    metrics_out = args.checkpoint / f"{dataset_name}_ft_qwen_metrics.json"

    # Save exactly like DINO does
    save_predictions_csv(
        build_prediction_rows(
            preds, labels, video_ids, frame_ids, frame_paths, widths, heights, task=args.task
        ),
        predictions_out,
    )

    metrics["checkpoint"] = str(args.checkpoint)
    metrics["predictions_csv"] = str(predictions_out)

    metrics_out.write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to: {metrics_out}")
    print(f"Saved predictions to: {predictions_out}")
    return metrics