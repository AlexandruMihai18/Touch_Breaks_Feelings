import numpy as np
import torch
from PIL import Image

from .models import (INFERENCE_SEED, device, processor, sam_model)



def extract_points_from_mask(mask, num_points=10):
    if mask.ndim == 3:
        mask = mask[..., 0]
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    rng = np.random.default_rng(INFERENCE_SEED)
    idx = rng.choice(len(xs), size=min(num_points, len(xs)), replace=False)
    return np.stack([xs[idx], ys[idx]], axis=-1).tolist()


def run_sam(image_np, scribble_mask):
    """Run SAM2 on image_np using scribble_mask to extract prompt points.

    Returns (binary_mask uint8, list_of_[x,y]_points).
    Used by the manual annotation app tab (scribble-driven workflow).
    """
    image = Image.fromarray(image_np).convert("RGB")

    alpha_or_gray = (
        scribble_mask[..., 3]
        if (scribble_mask.ndim == 3 and scribble_mask.shape[-1] == 4)
        else scribble_mask
    )
    points = extract_points_from_mask(alpha_or_gray)
    if points is None:
        return np.zeros(image_np.shape[:2], dtype=np.uint8), []

    inputs = processor(
        images=image,
        input_points=[[points]],
        input_labels=[[[1] * len(points)]],
        return_tensors="pt",
    )

    original_sizes = inputs.get("original_sizes")
    if original_sizes is None:
        w, h = image.size
        original_sizes = torch.tensor([[h, w]])

    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = sam_model(**inputs)

    masks = processor.post_process_masks(outputs.pred_masks, original_sizes.cpu())
    mask = masks[0][0].cpu().numpy()
    if mask.ndim == 3:
        mask = mask[1]

    return (mask > 0.0).astype(np.uint8) * 255, points
