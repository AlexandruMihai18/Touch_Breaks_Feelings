from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import numpy as np
import torch
from PIL import Image

from .types import PromptMaskResult
from .utils import binary_uint8_mask, normalize_text_prompt, zero_mask

_MODEL_ENV = "SAM3_MODEL_ID"
DEFAULT_SAM3_MODEL_ID = "facebook/sam3"


def run_sam3(
    image_np: np.ndarray,
    text: str,
    model_id: str | None = None,
    score_threshold: float = 0.0,
) -> PromptMaskResult:
    """Run a Hugging Face SAM3-style text-prompt segmentation backend.

    ``model_id`` can also be supplied through the ``SAM3_MODEL_ID`` environment
    variable. If neither is provided, ``facebook/sam3`` is used. The backend is
    loaded lazily and should expose either processor post-processing methods
    that return masks, or model outputs with mask fields such as
    ``pred_masks``/``masks`` and optional ``scores``/``boxes``.
    """
    original_prompt = normalize_text_prompt(text)
    resolved_model_id = model_id or os.environ.get(_MODEL_ENV) or DEFAULT_SAM3_MODEL_ID

    if not original_prompt:
        return PromptMaskResult(
            mask=zero_mask(image_np),
            prompt=original_prompt,
            score=0.0,
            box=[],
            backend="sam3",
            debug_image=None,
        )

    processor, model, device = _load_hf_sam3(resolved_model_id)
    image = Image.fromarray(image_np).convert("RGB")
    inputs = processor(images=image, text=original_prompt, return_tensors="pt")
    inputs = _move_to_device(inputs, device)

    with torch.inference_mode():
        outputs = model(**inputs)

    candidates = _extract_candidates(
        processor=processor,
        outputs=outputs,
        target_size=image_np.shape[:2],
        score_threshold=score_threshold,
    )
    if not candidates:
        return PromptMaskResult(
            mask=zero_mask(image_np),
            prompt=original_prompt,
            score=0.0,
            box=[],
            backend="sam3",
            debug_image=None,
        )

    best = max(candidates, key=lambda item: item["score"])
    return PromptMaskResult(
        mask=binary_uint8_mask(best["mask"], shape=image_np.shape[:2]),
        prompt=original_prompt,
        score=float(best["score"]),
        box=[float(v) for v in best.get("box", [])],
        backend="sam3",
        debug_image=None,
    )


@lru_cache(maxsize=4)
def _load_hf_sam3(model_id: str):
    from transformers import AutoModel, AutoProcessor

    trust_remote_code = os.environ.get("SAM3_TRUST_REMOTE_CODE", "1") not in {"0", "false", "False"}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=trust_remote_code).to(device)
    model.eval()
    return processor, model, device


def _move_to_device(inputs: Any, device: str) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    if isinstance(inputs, dict):
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    return inputs


def _extract_candidates(
    processor: Any,
    outputs: Any,
    target_size: tuple[int, int],
    score_threshold: float,
) -> list[dict[str, Any]]:
    processed = _post_process_candidates(processor, outputs, target_size)
    if processed is not None:
        return _candidates_from_processed(processed, score_threshold)
    return _candidates_from_outputs(outputs, score_threshold)


def _post_process_candidates(processor: Any, outputs: Any, target_size: tuple[int, int]) -> Any:
    for name in (
        "post_process_instance_segmentation",
        "post_process_semantic_segmentation",
        "post_process_panoptic_segmentation",
    ):
        fn = getattr(processor, name, None)
        if fn is None:
            continue
        try:
            return fn(outputs, target_sizes=[target_size])
        except TypeError:
            try:
                return fn(outputs, target_size=[target_size])
            except TypeError:
                continue
    return None


def _candidates_from_processed(processed: Any, score_threshold: float) -> list[dict[str, Any]]:
    if isinstance(processed, (list, tuple)) and processed:
        processed = processed[0]

    if isinstance(processed, dict):
        masks = _get_first_present(processed, ("masks", "pred_masks", "segmentation"))
        scores = _get_first_present(processed, ("scores", "confidence_scores"))
        boxes = _get_first_present(processed, ("boxes", "pred_boxes"))
        return _pack_candidates(masks, scores, boxes, score_threshold)

    return _pack_candidates(processed, None, None, score_threshold)


def _candidates_from_outputs(outputs: Any, score_threshold: float) -> list[dict[str, Any]]:
    masks = _get_attr_or_key(outputs, ("masks", "pred_masks", "mask_logits", "logits"))
    scores = _get_attr_or_key(outputs, ("scores", "confidence_scores", "pred_scores"))
    boxes = _get_attr_or_key(outputs, ("boxes", "pred_boxes"))
    return _pack_candidates(masks, scores, boxes, score_threshold)


def _pack_candidates(
    masks: Any,
    scores: Any,
    boxes: Any,
    score_threshold: float,
) -> list[dict[str, Any]]:
    if masks is None:
        return []

    masks_np = _to_numpy(masks)
    scores_np = _to_numpy(scores) if scores is not None else None
    boxes_np = _to_numpy(boxes) if boxes is not None else None

    if masks_np.ndim == 2:
        masks_np = masks_np[None, ...]
    if masks_np.ndim == 4 and masks_np.shape[0] == 1:
        masks_np = masks_np[0]

    candidates = []
    for idx, mask in enumerate(masks_np):
        score = float(scores_np[idx]) if scores_np is not None and idx < len(scores_np) else 1.0
        if score < score_threshold:
            continue
        box = boxes_np[idx].tolist() if boxes_np is not None and idx < len(boxes_np) else []
        candidates.append({"mask": mask, "score": score, "box": box})
    return candidates


def _get_first_present(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _get_attr_or_key(obj: Any, keys: tuple[str, ...]) -> Any:
    if isinstance(obj, dict):
        return _get_first_present(obj, keys)
    for key in keys:
        if hasattr(obj, key):
            return getattr(obj, key)
    return None


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)
