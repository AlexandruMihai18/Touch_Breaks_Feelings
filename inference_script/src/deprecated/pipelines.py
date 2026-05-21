import numpy as np
from PIL import Image

from ..sam import run_sam
from ..seggpt import run_seggpt
from ..visualization import draw_dual_mask_viz
from .touch import compute_touch_region_with_depth
from .visualization import draw_sam_viz, draw_seggpt_viz, draw_dual_seggpt_viz

from touch_detection_alg.depth import colorize_depth, predict_depth


def run_sam_and_seggpt(prompt_editor, img1, img2):
    """Single-scribble pipeline: SAM on prompt image → SegGPT on two target images.

    prompt_editor : dict from gr.ImageEditor with keys 'background' and 'layers'
    img1, img2    : numpy RGB arrays for the two target images
    Returns three overlay visualizations (prompt, img1, img2).
    """
    background = prompt_editor["background"]
    layers = prompt_editor["layers"]
    scribble = layers[0] if layers else np.zeros_like(background)

    sam_mask, points = run_sam(background, scribble)

    prompt_img_pil = Image.fromarray(background).convert("RGB")
    sam_mask_pil = Image.fromarray(sam_mask).convert("L")

    mask1 = run_seggpt(
        Image.fromarray(img1).convert("RGB"), prompt_img_pil, sam_mask_pil
    )
    mask2 = run_seggpt(
        Image.fromarray(img2).convert("RGB"), prompt_img_pil, sam_mask_pil
    )

    return (
        draw_sam_viz(background, sam_mask, points),
        draw_seggpt_viz(img1, mask1),
        draw_seggpt_viz(img2, mask2),
    )


def run_dual_sam_seggpt(prompt_editor, img1, img2):
    """Two-scribble pipeline: SAM per object → SegGPT propagation with class colors.

    prompt_editor : dict from gr.ImageEditor; layer 0 = obj1, layer 1 = obj2
    Returns prompt viz (class-colored SAM masks) + two target image vizs.
    """
    background = prompt_editor["background"]
    layers = prompt_editor["layers"]
    blank = np.zeros((*background.shape[:2], 4), dtype=np.uint8)
    scribble1 = layers[0] if len(layers) > 0 else blank
    scribble2 = layers[1] if len(layers) > 1 else blank

    sam_mask1, points1 = run_sam(background, scribble1)
    sam_mask2, points2 = run_sam(background, scribble2)

    h, w = background.shape[:2]
    label_map = np.zeros((h, w), dtype=np.uint8)
    label_map[sam_mask1 > 0] = 1
    label_map[sam_mask2 > 0] = 2

    prompt_img_pil = Image.fromarray(background).convert("RGB")

    pred1 = run_seggpt(
        Image.fromarray(img1).convert("RGB"), prompt_img_pil, label_map, num_labels=2
    )
    pred2 = run_seggpt(
        Image.fromarray(img2).convert("RGB"), prompt_img_pil, label_map, num_labels=2
    )

    def _split(pred):
        m1 = (pred == 1).astype(np.uint8) * 255
        m2 = (pred == 2).astype(np.uint8) * 255
        return m1, m2

    pred1_m1, pred1_m2 = _split(pred1)
    pred2_m1, pred2_m2 = _split(pred2)

    return (
        draw_dual_seggpt_viz(background, sam_mask1, sam_mask2, points1, points2),
        draw_dual_seggpt_viz(img1, pred1_m1, pred1_m2, points1, points2),
        draw_dual_seggpt_viz(img2, pred2_m1, pred2_m2, points1, points2),
    )


def run_touch_detection_with_depth(prompt_editor, dilation_radius, depth_threshold):
    """Touch detection pipeline augmented with depth-based contact filtering.

    Runs SAM on both scribble layers, estimates scene depth with Depth-Anything V2,
    then keeps only contact-zone pixels where both objects are at similar depth.

    Returns (touch_viz, depth_viz).
    """
    background = prompt_editor["background"]
    layers = prompt_editor["layers"]
    blank = np.zeros((*background.shape[:2], 4), dtype=np.uint8)
    scribble1 = layers[0] if len(layers) > 0 else blank
    scribble2 = layers[1] if len(layers) > 1 else blank

    mask1, points1 = run_sam(background, scribble1)
    mask2, points2 = run_sam(background, scribble2)

    depth_map = predict_depth(background)
    touch_mask = compute_touch_region_with_depth(
        mask1, mask2, depth_map, int(dilation_radius), float(depth_threshold)
    )

    touch_viz = draw_dual_mask_viz(
        background, mask1, mask2, touch_mask, points1, points2
    )
    depth_viz = colorize_depth(depth_map)
    return touch_viz, depth_viz


def run_video_touch_inference(
    prompt_editor, dilation_radius, depth_threshold, *sampled_frames
):
    """Two-scribble pipeline over a set of pre-sampled video frames with depth filtering.

    SAM runs once per scribble on the prompt image.
    SegGPT propagates each SAM mask to every sampled frame.
    Touch region is computed per frame using depth-based contact filtering.

    Returns [prompt_viz, frame1_viz, ..., frameN_viz].
    """
    background = prompt_editor["background"]
    layers = prompt_editor["layers"]
    blank = np.zeros((*background.shape[:2], 4), dtype=np.uint8)
    scribble1 = layers[0] if len(layers) > 0 else blank
    scribble2 = layers[1] if len(layers) > 1 else blank

    # SAM on prompt image — once per scribble, reused for all frames
    sam_mask1, points1 = run_sam(background, scribble1)
    sam_mask2, points2 = run_sam(background, scribble2)

    prompt_img_pil = Image.fromarray(background).convert("RGB")
    sam_mask1_pil = Image.fromarray(sam_mask1).convert("L")
    sam_mask2_pil = Image.fromarray(sam_mask2).convert("L")

    prompt_depth = predict_depth(background)
    prompt_touch = compute_touch_region_with_depth(
        sam_mask1, sam_mask2, prompt_depth, int(dilation_radius), float(depth_threshold)
    )
    prompt_viz = draw_dual_mask_viz(
        background, sam_mask1, sam_mask2, prompt_touch, points1, points2
    )

    frame_vizs = []
    for frame in sampled_frames:
        if frame is None:
            frame_vizs.append(None)
            continue
        frame_pil = Image.fromarray(frame).convert("RGB")
        frame_mask1 = run_seggpt(frame_pil, prompt_img_pil, sam_mask1_pil)
        frame_mask2 = run_seggpt(frame_pil, prompt_img_pil, sam_mask2_pil)
        frame_depth = predict_depth(frame)
        frame_touch = compute_touch_region_with_depth(
            frame_mask1,
            frame_mask2,
            frame_depth,
            int(dilation_radius),
            float(depth_threshold),
        )
        frame_vizs.append(
            draw_dual_mask_viz(frame, frame_mask1, frame_mask2, frame_touch, [], [])
        )

    return [prompt_viz] + frame_vizs
