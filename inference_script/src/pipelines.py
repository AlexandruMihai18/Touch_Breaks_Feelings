import numpy as np
from PIL import Image

from .sam import run_sam
from .seggpt import run_seggpt
from .visualization import draw_dual_mask_viz

from touch_detection_alg.depth import colorize_depth, predict_depth
from touch_detection_alg.touch import compute_touch_region, compute_touch_region_v2

# =========================
# Touch Detection Demo (v2)
# =========================

_DEMO_DILATION_RADIUS = 25
_DEMO_ABS_D_THRESHOLD = 0.15


def run_touch_demo(prompt_editor, query_img):
    """Step-by-step touch detection demo for context + query images.

    Context path: scribble → SAM masks → initial dilation zone → depth map → v2 touch.
    Query path  : SegGPT propagation from context SAM masks → same four steps.

    Returns 8 RGB arrays:
        ctx_sam_viz, ctx_init_viz, ctx_depth_viz, ctx_final_viz,
        qry_seggpt_viz, qry_init_viz, qry_depth_viz, qry_final_viz
    """
    background = prompt_editor["background"]
    layers = prompt_editor["layers"]
    blank_layer = np.zeros((*background.shape[:2], 4), dtype=np.uint8)
    scribble1 = layers[0] if len(layers) > 0 else blank_layer
    scribble2 = layers[1] if len(layers) > 1 else blank_layer

    # ── Context: SAM ─────────────────────────────────────────────────────────
    sam_mask1, points1 = run_sam(background, scribble1)
    sam_mask2, points2 = run_sam(background, scribble2)

    no_touch = np.zeros_like(sam_mask1)
    ctx_sam_viz = draw_dual_mask_viz(background, sam_mask1, sam_mask2, no_touch, points1, points2)

    ctx_init_touch = compute_touch_region(sam_mask1, sam_mask2, _DEMO_DILATION_RADIUS)
    ctx_init_viz = draw_dual_mask_viz(background, sam_mask1, sam_mask2, ctx_init_touch, points1, points2)

    ctx_depth = predict_depth(background)
    ctx_depth_viz = colorize_depth(ctx_depth)

    ctx_touch_v2 = compute_touch_region_v2(
        sam_mask1, sam_mask2, ctx_depth, _DEMO_DILATION_RADIUS, _DEMO_ABS_D_THRESHOLD
    )
    ctx_final_viz = draw_dual_mask_viz(background, sam_mask1, sam_mask2, ctx_touch_v2, points1, points2)

    # ── Query: SegGPT ─────────────────────────────────────────────────────────
    if query_img is None:
        blank_rgb = np.zeros((*background.shape[:2], 3), dtype=np.uint8)
        return ctx_sam_viz, ctx_init_viz, ctx_depth_viz, ctx_final_viz, blank_rgb, blank_rgb, blank_rgb, blank_rgb

    h, w = background.shape[:2]
    label_map = np.zeros((h, w), dtype=np.uint8)
    label_map[sam_mask1 > 0] = 1
    label_map[sam_mask2 > 0] = 2

    prompt_img_pil = Image.fromarray(background).convert("RGB")
    pred = run_seggpt(
        Image.fromarray(query_img).convert("RGB"), prompt_img_pil, label_map, num_labels=2
    )
    qry_mask1 = (pred == 1).astype(np.uint8) * 255
    qry_mask2 = (pred == 2).astype(np.uint8) * 255

    no_touch_qry = np.zeros_like(qry_mask1)
    qry_seggpt_viz = draw_dual_mask_viz(query_img, qry_mask1, qry_mask2, no_touch_qry, [], [])

    qry_init_touch = compute_touch_region(qry_mask1, qry_mask2, _DEMO_DILATION_RADIUS)
    qry_init_viz = draw_dual_mask_viz(query_img, qry_mask1, qry_mask2, qry_init_touch, [], [])

    qry_depth = predict_depth(query_img)
    qry_depth_viz = colorize_depth(qry_depth)

    qry_touch_v2 = compute_touch_region_v2(
        qry_mask1, qry_mask2, qry_depth, _DEMO_DILATION_RADIUS, _DEMO_ABS_D_THRESHOLD
    )
    qry_final_viz = draw_dual_mask_viz(query_img, qry_mask1, qry_mask2, qry_touch_v2, [], [])

    return ctx_sam_viz, ctx_init_viz, ctx_depth_viz, ctx_final_viz, qry_seggpt_viz, qry_init_viz, qry_depth_viz, qry_final_viz
