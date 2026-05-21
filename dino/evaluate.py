from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

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
) -> tuple[list[int], list[int], list[str], list[str], list[str]]:
    model.eval()
    preds: list[int] = []
    labels: list[int] = []
    datasets: list[str] = []
    video_ids: list[str] = []
    frame_ids: list[str] = []
    with torch.inference_mode():
        for batch in loader:
            pixel_values, batch_labels, batch_datasets, batch_video_ids, batch_frame_ids = batch
            logits = model(pixel_values.to(device))
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            labels.extend(batch_labels.cpu().tolist())
            datasets.extend(list(batch_datasets))
            video_ids.extend(str(v) for v in batch_video_ids)
            frame_ids.extend(str(f) for f in batch_frame_ids)
    return preds, labels, datasets, video_ids, frame_ids


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


def build_prediction_rows(
    preds: list[int],
    labels: list[int],
    video_ids: list[str],
    frame_ids: list[str],
) -> list[dict[str, int | str]]:
    return [
        {
            "pred": pred,
            "label": label,
            "video_id": video_id,
            "frame_id": frame_id,
        }
        for pred, label, video_id, frame_id in zip(preds, labels, video_ids, frame_ids)
    ]


def save_predictions_csv(rows: list[dict[str, int | str]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pred", "label", "video_id", "frame_id"])
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
    predictions_out: str | Path | None = None,
) -> dict:
    dataset = DinoFrameDataset(samples, processor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    state = torch.load(checkpoint, map_location="cpu")
    resolved_head_type = head_type or (
        state.get("head_type") if isinstance(state, dict) else None
    ) or "mlp"
    model = DinoClassifier(encoder, head_type=resolved_head_type).to(device)
    if isinstance(state, dict) and "head_state_dict" in state:
        model.head.load_state_dict(state["head_state_dict"])
    else:
        model.head.load_state_dict(state)
    preds, labels, datasets, video_ids, frame_ids = run_predictions(model, loader, device)
    if predictions_out is not None:
        save_predictions_csv(
            build_prediction_rows(preds, labels, video_ids, frame_ids),
            predictions_out,
        )
    return compute_metrics(preds, labels, datasets)


def save_metrics(metrics: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))
