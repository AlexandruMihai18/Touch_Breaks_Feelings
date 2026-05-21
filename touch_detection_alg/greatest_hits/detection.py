"""GH-specific mask detection: stick via Grounding-DINO + SAM2, object at stick tip.

Public interface
----------------
    detect_stick(frame)                → (mask, score, box)
    detect_object(frame, stick_mask)   → (mask, tip)

Lower-level helpers are exposed for callers that need intermediate results
(e.g. the app's debug-endpoints panel):

    run_sam_text(frame, text)              → (mask, box, score, sharpened)
    run_sam_points(frame, pos, neg)        → mask
    detect_hand_center(frame, stick_mask)  → (center, score)
    find_stick_endpoints(stick_mask, ...)  → (ep_a, ep_b, shaft, tip_is_ep_a)
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from ..depth import sharpen_image
from .models import INFERENCE_SEED, get_gdino, get_sam

_STICK_PROMPT = "thin wooden stick."
_MIN_OBJ_PIXELS = 300


# ---------------------------------------------------------------------------
# Generic SAM2 helpers
# ---------------------------------------------------------------------------

def run_sam_text(
    image_np: np.ndarray,
    text: str,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
) -> tuple[np.ndarray, list, float, np.ndarray]:
    """Detect an object by text prompt (Grounding DINO → SAM2 box prompt).

    Returns (binary_mask uint8, best_box [x1,y1,x2,y2], detection_score, sharpened_image_np).
    best_box is an empty list and score is 0.0 when nothing is detected.
    """
    sam_processor, sam_model = get_sam()
    gdino_processor, gdino_model = get_gdino()
    device = next(sam_model.parameters()).device

    image = sharpen_image(image_np)
    sharpened_np = np.array(image)

    gdino_inputs = gdino_processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.inference_mode():
        gdino_outputs = gdino_model(**gdino_inputs)

    results = gdino_processor.post_process_grounded_object_detection(
        gdino_outputs,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(image.height, image.width)],
    )
    boxes = results[0]["boxes"]
    scores = results[0]["scores"]

    if len(boxes) == 0:
        return np.zeros(image_np.shape[:2], dtype=np.uint8), [], 0.0, sharpened_np

    best_idx = int(scores.argmax())
    best_box = boxes[best_idx].cpu().numpy().tolist()
    best_score = float(scores[best_idx])

    sam_inputs = sam_processor(images=image, input_boxes=[[best_box]], return_tensors="pt")
    original_sizes = sam_inputs.get("original_sizes")
    if original_sizes is None:
        w, h = image.size
        original_sizes = torch.tensor([[h, w]])

    sam_inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in sam_inputs.items()}
    with torch.inference_mode():
        sam_outputs = sam_model(**sam_inputs)

    masks = sam_processor.post_process_masks(sam_outputs.pred_masks, original_sizes.cpu())
    mask = masks[0][0].cpu().numpy()
    if mask.ndim == 3:
        mask = mask[1]
    return (mask > 0.0).astype(np.uint8) * 255, best_box, best_score, sharpened_np


def run_sam_points(
    image_np: np.ndarray,
    pos_points: list,
    neg_points: list,
) -> np.ndarray:
    """Run SAM2 with explicit positive (1) and negative (0) point prompts.

    Returns a binary uint8 mask.
    """
    if not pos_points:
        return np.zeros(image_np.shape[:2], dtype=np.uint8)

    sam_processor, sam_model = get_sam()
    device = next(sam_model.parameters()).device

    image = Image.fromarray(image_np).convert("RGB")
    all_points = pos_points + neg_points
    all_labels = [1] * len(pos_points) + [0] * len(neg_points)

    inputs = sam_processor(
        images=image,
        input_points=[[all_points]],
        input_labels=[[all_labels]],
        return_tensors="pt",
    )
    original_sizes = inputs.get("original_sizes")
    if original_sizes is None:
        w, h = image.size
        original_sizes = torch.tensor([[h, w]])

    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = sam_model(**inputs)

    masks = sam_processor.post_process_masks(outputs.pred_masks, original_sizes.cpu())
    mask = masks[0][0].cpu().numpy()
    if mask.ndim == 3:
        mask = mask[1]
    return (mask > 0.0).astype(np.uint8) * 255


# ---------------------------------------------------------------------------
# Stick endpoint and tip detection
# ---------------------------------------------------------------------------

def find_stick_endpoints(
    stick_mask: np.ndarray,
    hand_center: list | None = None,
    hand_score: float = 0.0,
) -> tuple[list, list, list, bool]:
    """Find both endpoints and shaft sample points of a stick mask via PCA.

    Tip disambiguation uses a weighted blend of two signals:
    - Hand signal (weight = hand_score): the tip is the endpoint farther from
      the detected hand. Trusted more the higher the DINO confidence.
    - Heuristic signal (weight = 1 - hand_score): 70% proximity to image
      center + 30% distance from nearest border.
    When no hand is detected (hand_center is None) the heuristic runs alone.

    Returns (endpoint_a, endpoint_b, shaft_points, tip_is_ep_a).
    endpoint_a = argmax-projection end; endpoint_b = argmin end.
    Returns empty lists and False when the mask has too few pixels.
    """
    ys, xs = np.where(stick_mask > 0)
    if len(xs) < 20:
        return [], [], [], False

    coords = np.stack([xs, ys], axis=1).astype(float)
    centroid = coords.mean(axis=0)
    _, _, vt = np.linalg.svd(coords - centroid, full_matrices=False)
    principal = vt[0]
    projections = (coords - centroid) @ principal

    ep_a = coords[np.argmax(projections)]
    ep_b = coords[np.argmin(projections)]

    h, w = stick_mask.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    max_center_dist = np.sqrt(cx ** 2 + cy ** 2)
    max_border_dist = min(w, h) / 2.0

    def _heuristic_score(pt):
        center_dist = np.sqrt((pt[0] - cx) ** 2 + (pt[1] - cy) ** 2)
        border_dist = min(int(pt[0]), w - 1 - int(pt[0]), int(pt[1]), h - 1 - int(pt[1]))
        return 0.7 * (1 - center_dist / max_center_dist) + 0.3 * (border_dist / max_border_dist)

    h_score_a = _heuristic_score(ep_a)
    h_score_b = _heuristic_score(ep_b)

    if hand_center is not None:
        hx, hy = hand_center
        dist_a = np.sqrt((ep_a[0] - hx) ** 2 + (ep_a[1] - hy) ** 2)
        dist_b = np.sqrt((ep_b[0] - hx) ** 2 + (ep_b[1] - hy) ** 2)
        max_dist = max(dist_a, dist_b)
        hand_score_a = dist_a / max_dist if max_dist > 0 else 0.5
        hand_score_b = dist_b / max_dist if max_dist > 0 else 0.5

        w_hand = hand_score ** 1.3
        score_a = w_hand * hand_score_a + (1.0 - w_hand) * h_score_a
        score_b = w_hand * hand_score_b + (1.0 - w_hand) * h_score_b
        tip_is_ep_a = score_a >= score_b
    else:
        tip_is_ep_a = h_score_a >= h_score_b

    p20, p60 = np.percentile(projections, [20, 60])
    shaft_idx = np.where((projections >= p20) & (projections <= p60))[0]
    n = min(3, len(shaft_idx))
    sampled = coords[shaft_idx[np.linspace(0, len(shaft_idx) - 1, n, dtype=int)]]

    to_pt = lambda p: [int(p[0]), int(p[1])]
    return to_pt(ep_a), to_pt(ep_b), [to_pt(p) for p in sampled], tip_is_ep_a


def detect_hand_center(
    image_np: np.ndarray,
    stick_mask: np.ndarray | None = None,
    box_threshold: float = 0.4,
    text_threshold: float = 0.4,
    min_score: float = 0.4,
    max_bbox_area_frac: float = 0.20,
    max_stick_fill_frac: float = 0.40,
) -> tuple[list, float] | tuple[None, float]:
    """Run Grounding DINO to detect a hand in the frame.

    Returns (center [cx, cy], score). Returns (None, 0.0) when no detection
    passes the filters (size, confidence, stick-overlap).
    """
    gdino_processor, gdino_model = get_gdino()
    device = next(gdino_model.parameters()).device

    image = Image.fromarray(image_np).convert("RGB")
    inputs = gdino_processor(images=image, text="a hand.", return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = gdino_model(**inputs)

    results = gdino_processor.post_process_grounded_object_detection(
        outputs,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(image.height, image.width)],
    )
    boxes = results[0]["boxes"]
    scores = results[0]["scores"]
    if len(boxes) == 0:
        return None, 0.0

    best_idx = int(scores.argmax())
    best_score = float(scores[best_idx])
    if best_score < min_score:
        return None, 0.0

    x1, y1, x2, y2 = boxes[best_idx].cpu().numpy()
    H, W = image_np.shape[:2]
    if (x2 - x1) * (y2 - y1) / (W * H) > max_bbox_area_frac:
        return None, 0.0

    if stick_mask is not None:
        xi1, yi1, xi2, yi2 = int(x1), int(y1), int(x2), int(y2)
        box_area = max(1, (xi2 - xi1) * (yi2 - yi1))
        if int(np.sum(stick_mask[yi1:yi2, xi1:xi2] > 0)) / box_area > max_stick_fill_frac:
            return None, 0.0

    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0], best_score


def _tip_point_cloud(tip: list, stick_mask: np.ndarray, n_points: int = 5, radius: int = 15) -> list:
    """Sample N positive SAM points in a tight ring around the tip, outside the stick."""
    x, y = tip
    h, w = stick_mask.shape[:2]

    try:
        import cv2
        exclusion = cv2.dilate(stick_mask, np.ones((7, 7), np.uint8))
    except ImportError:
        exclusion = stick_mask

    candidates = []
    for r in [max(1, radius // 2), radius]:
        for angle in np.linspace(0, 2 * np.pi, 24, endpoint=False):
            cx = int(round(x + r * np.cos(angle)))
            cy = int(round(y + r * np.sin(angle)))
            if 0 <= cx < w and 0 <= cy < h and exclusion[cy, cx] == 0:
                candidates.append([cx, cy])

    if not candidates:
        return [tip]

    idx = np.round(
        np.linspace(0, len(candidates) - 1, min(n_points, len(candidates)))
    ).astype(int)
    return [candidates[i] for i in idx]


def run_sam_from_stick_tip(
    image_np: np.ndarray,
    stick_mask: np.ndarray,
) -> tuple[np.ndarray, list]:
    """Segment the object at the stick tip.

    Tip selection: DINO first tries to detect a hand — when found, the tip is
    the endpoint farthest from the hand (the hand marks the handle end). Falls
    back to a weighted heuristic when no hand is detected.

    Positive prompts: a tight point cloud around the tip, strictly outside the
    stick mask. Negative prompts: handle endpoint + shaft samples.

    Returns (object_mask uint8, tip_point [x, y]).
    """
    hand_center, hand_score = detect_hand_center(image_np, stick_mask)
    ep_a, ep_b, shaft_points, tip_is_ep_a = find_stick_endpoints(stick_mask, hand_center, hand_score)
    if not ep_a:
        return np.zeros(image_np.shape[:2], dtype=np.uint8), []

    tip    = ep_a if tip_is_ep_a else ep_b
    handle = ep_b if tip_is_ep_a else ep_a

    sharpened = np.array(sharpen_image(image_np))
    cloud = _tip_point_cloud(tip, stick_mask)
    mask  = run_sam_points(sharpened, cloud, [handle] + shaft_points)
    return mask, tip


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def detect_stick(frame: np.ndarray) -> tuple[np.ndarray, float, list]:
    """Detect the drum stick in a frame using Grounding-DINO + SAM2.

    Returns (mask uint8, detection_score, best_box).
    score is 0.0 and mask is all-zero when nothing is detected.
    """
    mask, box, score, _ = run_sam_text(frame, _STICK_PROMPT)
    return mask, score, box


def detect_object(frame: np.ndarray, stick_mask: np.ndarray) -> tuple[np.ndarray, list]:
    """Segment the object at the stick tip.

    Returns (mask uint8, tip [x, y]).
    tip is an empty list when detection fails.
    """
    return run_sam_from_stick_tip(frame, stick_mask)
