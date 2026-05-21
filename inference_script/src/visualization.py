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



# ----- dual-mask visualization (used by touch tabs) -----


def draw_stick_viz(image_np, stick_mask, box=None, color=(255, 220, 0)):
    """Overlay a detection mask on image with bounding box. Yellow by default (stick)."""
    r, g, b = color
    base = _apply_overlay(
        Image.fromarray(image_np).convert("RGBA"), stick_mask, [r, g, b, 150]
    )
    vis = np.array(base.convert("RGB"))
    if box and _has_cv2:
        import cv2
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
    return vis


def draw_tip_detection_viz(image_np, stick_mask, obj_mask, tip_point=None):
    """Yellow = stick, blue = object at tip, red dot = tip prompt point."""
    base = Image.fromarray(image_np).convert("RGBA")
    base = _apply_overlay(base, stick_mask, [255, 220, 0, 120])
    base = _apply_overlay(base, obj_mask, [30, 120, 255, 150])
    vis = np.array(base.convert("RGB"))
    if tip_point:
        vis = _draw_dots(vis, [tip_point], (255, 50, 50), radius=8)
    return vis


def draw_endpoints_debug_viz(
    image_np: np.ndarray,
    stick_mask: np.ndarray,
    ep_a: list,
    ep_b: list,
    tip_is_ep_a: bool,
    hand_center: list | None = None,
) -> np.ndarray:
    """Debug viz for stick endpoint detection.

    Yellow overlay = stick mask.
    Green dot + 'A' = ep_a (PCA argmax end).
    Red dot   + 'B' = ep_b (PCA argmin end).
    White ring + 'TIP' = whichever endpoint was chosen as the tip.
    Yellow dot + 'HAND' = Grounding DINO hand center (when detected).
    """
    base = _apply_overlay(
        Image.fromarray(image_np).convert("RGBA"), stick_mask, [255, 220, 0, 120]
    )
    vis = np.array(base.convert("RGB"))
    if not ep_a:
        return vis
    if not _has_cv2:
        vis = _draw_dots(vis, [ep_a], (50, 220, 50))
        vis = _draw_dots(vis, [ep_b], (220, 50, 50))
        return vis
    import cv2
    tip = ep_a if tip_is_ep_a else ep_b

    ax, ay = int(ep_a[0]), int(ep_a[1])
    bx, by = int(ep_b[0]), int(ep_b[1])
    tx, ty = int(tip[0]), int(tip[1])

    cv2.circle(vis, (ax, ay), 10, (0, 0, 0), -1)
    cv2.circle(vis, (ax, ay), 8, (50, 220, 50), -1)
    cv2.putText(vis, "A", (ax + 12, ay + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 50), 2)

    cv2.circle(vis, (bx, by), 10, (0, 0, 0), -1)
    cv2.circle(vis, (bx, by), 8, (220, 50, 50), -1)
    cv2.putText(vis, "B", (bx + 12, by + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 50, 50), 2)

    cv2.circle(vis, (tx, ty), 18, (255, 255, 255), 2)
    cv2.putText(vis, "TIP", (tx + 20, ty - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if hand_center:
        hx, hy = int(hand_center[0]), int(hand_center[1])
        cv2.circle(vis, (hx, hy), 14, (0, 0, 0), -1)
        cv2.circle(vis, (hx, hy), 12, (255, 220, 0), -1)
        cv2.putText(vis, "HAND", (hx + 15, hy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 0), 2)

    return vis


def draw_dual_mask_viz(image_np, mask1, mask2, touch_mask, points1, points2):
    """Composite view showing both objects and their touch region.

    blue    = object 1 mask
    orange  = object 2 mask
    magenta = touch region (drawn last so it's always visible)
    cyan dots   = object 1 prompt points
    orange dots = object 2 prompt points
    """
    base = Image.fromarray(image_np).convert("RGBA")
    base = _apply_overlay(base, mask1, [30, 120, 255, 110])
    base = _apply_overlay(base, mask2, [255, 140, 30, 110])
    base = _apply_overlay(base, touch_mask, [220, 30, 220, 170])
    vis = np.array(base.convert("RGB"))
    vis = _draw_dots(vis, points1, (0, 200, 255))
    vis = _draw_dots(vis, points2, (255, 165, 0))
    return vis
