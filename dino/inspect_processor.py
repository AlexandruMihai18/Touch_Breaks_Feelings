from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .hf import auth_kwargs, explain_hf_load_error
from .train import DEFAULT_MODEL_ID


def _as_list(value) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return [float(value)]


def _jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if hasattr(value, "items"):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _tensor_stats(pixel_values: torch.Tensor) -> dict:
    tensor = pixel_values.detach().cpu()
    channel_tensor = tensor[0] if tensor.ndim == 4 else tensor
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "global": {
            "min": float(tensor.min()),
            "max": float(tensor.max()),
            "mean": float(tensor.mean()),
            "std": float(tensor.std()),
        },
        "per_channel": [
            {
                "min": float(channel.min()),
                "max": float(channel.max()),
                "mean": float(channel.mean()),
                "std": float(channel.std()),
            }
            for channel in channel_tensor
        ],
    }


def _denormalize(pixel_values: torch.Tensor, processor) -> Image.Image:
    tensor = pixel_values.detach().cpu()
    if tensor.ndim == 4:
        tensor = tensor[0]

    mean = _as_list(getattr(processor, "image_mean", None)) or [0.0, 0.0, 0.0]
    std = _as_list(getattr(processor, "image_std", None)) or [1.0, 1.0, 1.0]
    if len(mean) == 1:
        mean *= 3
    if len(std) == 1:
        std *= 3

    mean_t = torch.tensor(mean, dtype=tensor.dtype).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=tensor.dtype).view(-1, 1, 1)
    image = (tensor * std_t + mean_t).clamp(0, 1)
    arr = (image.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _preview_path(path: str | Path) -> Path:
    preview_path = Path(path).expanduser()
    if preview_path.exists() and preview_path.is_dir():
        return preview_path / "processor_preview.png"
    if preview_path.suffix == "":
        return preview_path.with_suffix(".png")
    return preview_path


def inspect_image(args: argparse.Namespace) -> dict:
    from transformers import AutoImageProcessor

    image_path = Path(args.image).expanduser()
    image = Image.open(image_path).convert("RGB")
    try:
        processor = AutoImageProcessor.from_pretrained(
            args.model_id,
            **auth_kwargs(args.hf_token),
        )
    except OSError as exc:
        raise explain_hf_load_error(exc, args.model_id) from exc
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]

    report = {
        "image_path": str(image_path),
        "model_id": args.model_id,
        "original_image": {
            "mode": image.mode,
            "size": list(image.size),
        },
        "processor": {
            "class": processor.__class__.__name__,
            "do_resize": _jsonable(getattr(processor, "do_resize", None)),
            "size": _jsonable(getattr(processor, "size", None)),
            "crop_size": _jsonable(getattr(processor, "crop_size", None)),
            "resample": str(getattr(processor, "resample", None)),
            "do_rescale": _jsonable(getattr(processor, "do_rescale", None)),
            "rescale_factor": _jsonable(getattr(processor, "rescale_factor", None)),
            "do_normalize": _jsonable(getattr(processor, "do_normalize", None)),
            "image_mean": _jsonable(getattr(processor, "image_mean", None)),
            "image_std": _jsonable(getattr(processor, "image_std", None)),
        },
        "pixel_values": _tensor_stats(pixel_values),
    }

    if args.preview_out is not None:
        preview_path = _preview_path(args.preview_out)
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        _denormalize(pixel_values, processor).save(preview_path)
        report["preview_out"] = str(preview_path)

    if args.report_out is not None:
        report_path = Path(args.report_out).expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect what AutoImageProcessor feeds into the DINO model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", required=True, type=Path, help="Input image path.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional Hugging Face token for gated/private model checkpoints.",
    )
    parser.add_argument(
        "--preview-out",
        type=Path,
        default=None,
        help="Optional path to save a de-normalized RGB preview of pixel_values.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=None,
        help="Optional path to save the JSON inspection report.",
    )
    return parser.parse_args()


def main() -> None:
    inspect_image(parse_args())


if __name__ == "__main__":
    main()
