from __future__ import annotations

import numpy as np

from .types import PromptMaskResult
from .utils import binary_uint8_mask, normalize_grounding_prompt, normalize_text_prompt, zero_mask


def run_grounded_sam2(
    image_np: np.ndarray,
    text: str,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
) -> PromptMaskResult:
    """Run text-prompt segmentation via Grounding DINO + SAM2.

    Returns a binary uint8 mask in the same 0/255 format as the existing
    ``touch_detection_alg.greatest_hits.detection.run_sam_text`` helper.
    """
    original_prompt = normalize_text_prompt(text)
    if not original_prompt:
        return PromptMaskResult(
            mask=zero_mask(image_np),
            prompt=original_prompt,
            score=0.0,
            box=[],
            backend="grounded_sam2",
            debug_image=None,
        )

    from touch_detection_alg.greatest_hits.detection import run_sam_text

    mask, box, score, debug_image = run_sam_text(
        image_np,
        normalize_grounding_prompt(original_prompt),
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    if score == 0.0 or not box:
        mask = zero_mask(image_np)
    else:
        mask = binary_uint8_mask(mask, shape=image_np.shape[:2])

    return PromptMaskResult(
        mask=mask,
        prompt=original_prompt,
        score=float(score),
        box=[float(v) for v in box],
        backend="grounded_sam2",
        debug_image=debug_image,
    )

