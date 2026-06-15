from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .types import PromptMaskResult


def normalize_text_prompt(text: str) -> str:
    return text.strip()


def normalize_grounding_prompt(text: str) -> str:
    prompt = normalize_text_prompt(text)
    if prompt and not prompt.endswith("."):
        prompt += "."
    return prompt


def binary_uint8_mask(mask: np.ndarray, shape: tuple[int, int] | None = None) -> np.ndarray:
    """Convert model mask output to a 2-D uint8 mask with values 0 and 255."""
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = _pick_2d_mask(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D mask, got shape {arr.shape}")

    out = (arr > 0).astype(np.uint8) * 255
    if shape is not None and out.shape != shape:
        raise ValueError(f"Expected mask shape {shape}, got {out.shape}")
    return out


def zero_mask(image_np: np.ndarray) -> np.ndarray:
    return np.zeros(np.asarray(image_np).shape[:2], dtype=np.uint8)


def build_label_map(results: Iterable[PromptMaskResult]) -> np.ndarray:
    """Build a SegGPT class-label prompt map from binary prompt mask results.

    Background is 0. Valid non-empty masks are assigned labels 1..N in input order.
    Empty masks are skipped, so labels remain compact.
    """
    label_map: np.ndarray | None = None
    next_label = 1

    for result in results:
        mask = binary_uint8_mask(result.mask)
        if label_map is None:
            label_map = np.zeros(mask.shape, dtype=np.uint8)
        elif mask.shape != label_map.shape:
            raise ValueError(
                f"All masks must have the same shape; got {mask.shape} and {label_map.shape}"
            )

        if np.any(mask > 0):
            if next_label > np.iinfo(np.uint8).max:
                raise ValueError("Too many masks for a uint8 label map")
            label_map[mask > 0] = next_label
            next_label += 1

    if label_map is None:
        raise ValueError("Cannot build a label map from no results")
    return label_map


def _pick_2d_mask(mask: np.ndarray) -> np.ndarray:
    if mask.shape[0] == 1:
        return mask[0]
    if mask.shape[-1] == 1:
        return mask[..., 0]
    if mask.shape[0] <= 8:
        return mask[-1]
    if mask.shape[-1] <= 8:
        return mask[..., -1]
    raise ValueError(f"Cannot infer a 2-D mask from shape {mask.shape}")

