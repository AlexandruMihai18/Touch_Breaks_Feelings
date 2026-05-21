import numpy as np
from PIL import Image

try:
    import cv2

    _has_cv2 = True
except ImportError:
    _has_cv2 = False


# ----- low-level primitives -----


def _apply_overlay(base_rgba, mask, color_rgba):
    """Alpha-composite a solid color_rgba region onto base_rgba wherever mask > 0."""
    overlay = np.zeros((*mask.shape[:2], 4), dtype=np.uint8)
    overlay[mask > 0] = color_rgba
    return Image.alpha_composite(base_rgba, Image.fromarray(overlay, mode="RGBA"))


def _draw_dots(vis, points, inner_color, radius=6):
    """Draw filled circles (with a dark border) at each [x, y] point in-place."""
    for x, y in points:
        if _has_cv2:
            cv2.circle(vis, (int(x), int(y)), radius + 2, (0, 0, 0), -1)
            cv2.circle(vis, (int(x), int(y)), radius, inner_color, -1)
        else:
            r = radius
            y0, y1 = max(0, int(y) - r), min(vis.shape[0], int(y) + r + 1)
            x0, x1 = max(0, int(x) - r), min(vis.shape[1], int(x) + r + 1)
            vis[y0:y1, x0:x1] = inner_color
    return vis


# ----- single-mask visualizations -----


def draw_sam_viz(image_np, sam_mask, points):
    """Overlay the SAM mask (green) on the image and draw red prompt dots."""
    base = _apply_overlay(
        Image.fromarray(image_np).convert("RGBA"), sam_mask, [0, 200, 80, 120]
    )
    vis = np.array(base.convert("RGB"))
    return _draw_dots(vis, points, (255, 50, 50))


def draw_seggpt_viz(image_np, seggpt_label_map):
    """Overlay the SegGPT predicted mask (green) on the image."""
    binary = (seggpt_label_map > 0).astype(np.uint8) * 255
    base = _apply_overlay(
        Image.fromarray(image_np).convert("RGBA"), binary, [0, 200, 80, 140]
    )
    return np.array(base.convert("RGB"))


def draw_dual_seggpt_viz(image_np, mask1, mask2, points1, points2):
    """Class-colored overlay: blue = obj1, orange = obj2."""
    base = Image.fromarray(image_np).convert("RGBA")
    base = _apply_overlay(base, mask1, [30, 120, 255, 130])
    base = _apply_overlay(base, mask2, [255, 140, 30, 130])
    vis = np.array(base.convert("RGB"))
    vis = _draw_dots(vis, points1, (0, 200, 255))
    vis = _draw_dots(vis, points2, (255, 165, 0))
    return vis
