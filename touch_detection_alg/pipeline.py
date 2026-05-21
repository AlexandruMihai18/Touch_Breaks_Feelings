"""Single-call interface for the full touch detection pipeline.

Combines depth estimation (predict_depth + optional guided sharpening) with
contact-zone filtering (compute_touch_region_v2) into cohesive helpers.

Import from here instead of wiring the individual steps manually:

    from touch_detection_alg.pipeline import annotate_touch

For callers that need the depth map separately (e.g. to save a depth PNG),
use compute_depth + compute_touch_region_v2 directly:

    from touch_detection_alg.pipeline import compute_depth
    from touch_detection_alg.touch import compute_touch_region_v2

    depth = compute_depth(image, ...)
    touch = compute_touch_region_v2(mask1, mask2, depth, ...)
"""

import numpy as np

from touch_detection_alg.depth import predict_depth, sharpen_depth
from touch_detection_alg.touch import compute_touch_region_v2


def compute_depth(
    image: np.ndarray,
    *,
    clahe_clip: float = 0.0,
    guided_filter: bool = True,
    refine_radius: int = 4,
    refine_eps: float = 0.1,
) -> np.ndarray:
    """Estimate and optionally sharpen depth for an RGB image.

    Parameters
    ----------
    image         : uint8 (H, W, 3) RGB frame
    clahe_clip    : CLAHE clip limit applied before estimation; 0 = off
    guided_filter : sharpen depth with guided image filter after estimation
    refine_radius : guided-filter spatial radius in pixels
    refine_eps    : guided-filter regularisation ε

    Returns
    -------
    float32 (H, W) depth map in [0, 1]
    """
    depth = predict_depth(image, clahe_clip=clahe_clip)
    if guided_filter:
        depth = sharpen_depth(depth, image, radius=refine_radius, eps=refine_eps)
    return depth


def annotate_touch(
    image: np.ndarray,
    mask1: np.ndarray,
    mask2: np.ndarray,
    *,
    dilation: int = 10,
    abs_d_threshold: float = 0.05,
    local_radius: int = 10,
    clahe_clip: float = 0.0,
    guided_filter: bool = True,
    refine_radius: int = 4,
    refine_eps: float = 0.1,
) -> np.ndarray:
    """Given an RGB image and two object masks, compute the depth-filtered touch mask.

    Parameters
    ----------
    image           : uint8 (H, W, 3) RGB frame
    mask1           : uint8 (H, W) — first mask (hand / stick)
    mask2           : uint8 (H, W) — second mask (object)
    dilation        : contact zone dilation radius in pixels
    abs_d_threshold : max |depth1 - depth2| [0, 1] to keep a contact pixel;
                      0.05 = strict, 0.20 = lenient
    local_radius    : box-window half-width for per-pixel depth estimates
    clahe_clip      : CLAHE clip limit applied before depth estimation; 0 = off
    guided_filter   : sharpen depth with guided image filter
    refine_radius   : guided-filter spatial radius
    refine_eps      : guided-filter regularisation ε

    Returns
    -------
    uint8 (H, W) touch mask with values 0 or 255
    """
    depth = compute_depth(
        image,
        clahe_clip=clahe_clip,
        guided_filter=guided_filter,
        refine_radius=refine_radius,
        refine_eps=refine_eps,
    )
    return compute_touch_region_v2(
        mask1, mask2, depth,
        dilation_radius=dilation,
        abs_d_threshold=abs_d_threshold,
        local_radius=local_radius,
    )
