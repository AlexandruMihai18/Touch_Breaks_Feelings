from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PromptMaskResult:
    """Text-prompt segmentation result compatible with SegGPT prompt masks.

    Example:
        from PIL import Image

        prompt_mask_pil = Image.fromarray(result.mask).convert("L")
    """

    mask: np.ndarray
    prompt: str
    score: float
    box: list[float]
    backend: str
    debug_image: np.ndarray | None = None

