from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from .dataset import DinoFrameDataset, DinoSample
from .model import DinoClassifier


def device_from_args(device_name: str = "auto") -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_predictions(
    model: DinoClassifier,
    loader: DataLoader,
    device: torch.device,
    task: str = "binary",
) -> tuple[list, list, list[str], list[str], list[str], list[str], list[float], list[float]]:
    model.eval()
    preds: list = []
    labels: list = []
    datasets: list[str] = []
    video_ids: list[str] = []
    frame_ids: list[str] = []
    frame_paths: list[str] = []
    widths: list[float] = []
    heights: list[float] = []
    with torch.inference_mode():
        for batch in loader:
            (
                pixel_values,
                batch_labels,
                batch_datasets,
                batch_video_ids,
                batch_frame_ids,
                batch_frame_paths,
                batch_widths,
                batch_heights,
            ) = batch
            logits = model(pixel_values.to(device))
            if task == "point":
                preds.extend(logits.detach().cpu().tolist())
                labels.extend(batch_labels.cpu().tolist())
            else:
                preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
                labels.extend(batch_labels.cpu().tolist())
            datasets.extend(list(batch_datasets))
            video_ids.extend(str(v) for v in batch_video_ids)
            frame_ids.extend(str(f) for f in batch_frame_ids)
            frame_paths.extend(str(p) for p in batch_frame_paths)
            widths.extend(float(w) for w in batch_widths.cpu().tolist())
            heights.extend(float(h) for h in batch_heights.cpu().tolist())
    return preds, labels, datasets, video_ids, frame_ids, frame_paths, widths, heights


def compute_metrics(preds: list[int], labels: list[int], datasets: list[str]) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0,
    )
    matrix = confusion_matrix(labels, preds, labels=[0, 1]).tolist()
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "confusion_matrix": {
            "labels": ["no-touch", "touch"],
            "matrix": matrix,
        },
        "num_samples": len(labels),
        "class_counts": dict(Counter(labels)),
        "dataset_counts": dict(Counter(datasets)),
    }


def compute_point_metrics(
    preds: list[list[float]],
    labels: list[list[float]],
    datasets: list[str],
    widths: list[float],
    heights: list[float],
) -> dict:
    pred_arr = np.asarray(preds, dtype=np.float64)
    label_arr = np.asarray(labels, dtype=np.float64)
    diff = pred_arr - label_arr
    abs_diff = np.abs(diff)
    euclidean = np.linalg.norm(diff, axis=1)

    sizes = np.asarray(list(zip(widths, heights)), dtype=np.float64)
    pixel_diff = diff * np.maximum(sizes - 1.0, 1.0)
    pixel_euclidean = np.linalg.norm(pixel_diff, axis=1)

    return {
        "mean_euclidean_error": float(euclidean.mean()),
        "median_euclidean_error": float(np.median(euclidean)),
        "mean_abs_x_error": float(abs_diff[:, 0].mean()),
        "mean_abs_y_error": float(abs_diff[:, 1].mean()),
        "mean_pixel_euclidean_error": float(pixel_euclidean.mean()),
        "median_pixel_euclidean_error": float(np.median(pixel_euclidean)),
        "num_samples": len(labels),
        "dataset_counts": dict(Counter(datasets)),
    }


def prediction_csv_name(
    dataset_name: str,
    task: str,
    head_type: str,
    mlp_layers: int | None = None,
) -> str:
    head_name = f"mlp{mlp_layers}" if head_type == "mlp" and mlp_layers is not None else head_type
    return f"{dataset_name}_dino_{task}_{head_name}_prediction.csv"


def dataset_output_name(datasets: list[str]) -> str:
    unique = sorted(set(datasets))
    return unique[0] if len(unique) == 1 else "mixed"


def build_prediction_rows(
    preds: list,
    labels: list,
    video_ids: list[str],
    frame_ids: list[str],
    frame_paths: list[str],
    widths: list[float],
    heights: list[float],
    task: str,
) -> list[dict[str, int | float | str]]:
    rows: list[dict[str, int | float | str]] = []
    for pred, label, video_id, frame_id, frame_path, width, height in zip(
        preds, labels, video_ids, frame_ids, frame_paths, widths, heights
    ):
        if task == "point":
            x_pred = float(pred[0]) * max(float(width) - 1.0, 1.0)
            y_pred = float(pred[1]) * max(float(height) - 1.0, 1.0)
            rows.append(
                {
                    "frame_id": frame_id,
                    "video_id": video_id,
                    "frame_path": frame_path,
                    "audio_path": "NaN",
                    "label": "NaN",
                    "prediction": "NaN",
                    "x_touch": x_pred,
                    "y_touch": y_pred,
                }
            )
        else:
            rows.append(
                {
                    "frame_id": frame_id,
                    "video_id": video_id,
                    "frame_path": frame_path,
                    "audio_path": "NaN",
                    "label": int(label),
                    "prediction": int(pred),
                    "x_touch": "NaN",
                    "y_touch": "NaN",
                }
            )
    return rows


def save_predictions_csv(rows: list[dict[str, int | float | str]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_id",
                "video_id",
                "frame_path",
                "audio_path",
                "label",
                "prediction",
                "x_touch",
                "y_touch",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def evaluate_samples(
    samples: list[DinoSample],
    processor,
    encoder,
    checkpoint: str | Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    head_type: str | None = None,
    mlp_layers: int | None = None,
    predictions_out: str | Path | None = None,
    task: str | None = None,
) -> dict:
    dataset = DinoFrameDataset(samples, processor, task=task)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    state = torch.load(checkpoint, map_location="cpu")
    resolved_head_type = head_type or (
        state.get("head_type") if isinstance(state, dict) else None
    ) or "mlp"
    checkpoint_mlp_layers = state.get("mlp_layers") if isinstance(state, dict) else None
    resolved_mlp_layers = mlp_layers if mlp_layers is not None else checkpoint_mlp_layers
    if resolved_mlp_layers is None:
        resolved_mlp_layers = 2
    if resolved_head_type == "mlp" and resolved_mlp_layers < 2:
        raise ValueError("MLP heads need at least 2 linear layers")
    resolved_task = task or (state.get("task") if isinstance(state, dict) else None) or "binary"
    model = DinoClassifier(
        encoder,
        head_type=resolved_head_type,
        mlp_layers=resolved_mlp_layers,
        num_classes=2,
    ).to(device)
    if isinstance(state, dict) and "head_state_dict" in state:
        model.head.load_state_dict(state["head_state_dict"])
    else:
        model.head.load_state_dict(state)
    preds, labels, datasets, video_ids, frame_ids, frame_paths, widths, heights = run_predictions(
        model, loader, device, task=resolved_task
    )
    if predictions_out is not None:
        save_predictions_csv(
            build_prediction_rows(
                preds,
                labels,
                video_ids,
                frame_ids,
                frame_paths,
                widths,
                heights,
                task=resolved_task,
            ),
            predictions_out,
        )
    if resolved_task == "point":
        return compute_point_metrics(preds, labels, datasets, widths, heights)
    return compute_metrics(preds, labels, datasets)


def save_metrics(metrics: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))
