import cv2
import numpy as np

def _depth_adaptive_radius(base_radius, avg_depth, window_factor):
    """Window radius scaled inversely with depth: near objects get a larger window.

    scale maps depth [0, 1] → factor [1.0, 0.25] so a nearby object (depth≈0)
    gets a window up to window_factor × base_radius, while a distant one
    (depth≈1) gets window_factor × 0.25 × base_radius.  Always >= base_radius.
    """
    scale = 1.0 - 0.75 * float(np.clip(avg_depth, 0.0, 1.0))
    return max(base_radius, int(window_factor * base_radius * scale))


def compute_touch_region_with_depth(
    mask1, mask2, depth_map, dilation_radius=10, depth_threshold=0.3, window_factor=2.0
):
    """Contact zone gated by near-face relative-depth similarity.

    Two dilation radii are used:
    - dilation_radius : tight radius that defines the displayed contact zone.
    - depth-adaptive window (>= dilation_radius) : larger radius used only to
      sample near-face pixels for the depth comparison.  Each object's window
      is scaled inversely with its own mean depth so that a closer (larger in
      image space) object gets a proportionally bigger sampling window —
      a rough perspective correction.

    Depth test uses a *relative* difference normalised by the closer object's
    depth: gap / min(avg1, avg2).  This makes depth_threshold scale-invariant —
    e.g. 0.3 means "the farther object is at most 30% deeper than the closer
    one" — and naturally tightens the test for close objects (where the same
    pixel-depth gap is more significant) while relaxing it for distant ones.

    Bootstrapping order:
      1. Compute each object's full-mask mean depth (cheap global estimate).
      2. Derive each object's adaptive window radius from that depth.
      3. mask1_near = mask1 pixels inside mask2's window (d2_win).
         mask2_near = mask2 pixels inside mask1's window (d1_win).
      4. Compare near-face relative depths; suppress contact zone if too different.
    """
    d1 = _dilate_mask(mask1, dilation_radius)
    d2 = _dilate_mask(mask2, dilation_radius)
    contact_zone = d1 & d2

    if not contact_zone.any():
        return np.zeros_like(mask1)

    if not (mask1 > 0).any() or not (mask2 > 0).any():
        return contact_zone.astype(np.uint8) * 255

    depth_norm = depth_map.astype(np.float32)
    d_min, d_max = depth_norm.min(), depth_norm.max()
    if d_max > d_min:
        depth_norm = (depth_norm - d_min) / (d_max - d_min)

    # Step 1: full-mask depth → adaptive window radius per object
    avg_depth1 = float(depth_norm[mask1 > 0].mean())
    avg_depth2 = float(depth_norm[mask2 > 0].mean())

    d1_win = _dilate_mask(
        mask1, _depth_adaptive_radius(dilation_radius, avg_depth1, window_factor)
    )
    d2_win = _dilate_mask(
        mask2, _depth_adaptive_radius(dilation_radius, avg_depth2, window_factor)
    )

    # Step 2: near-face pixels of each object
    mask1_near = (mask1 > 0) & d2_win  # mask1 pixels within mask2's reach
    mask2_near = (mask2 > 0) & d1_win  # mask2 pixels within mask1's reach

    if not mask1_near.any() or not mask2_near.any():
        return contact_zone.astype(np.uint8) * 255

    avg1_near = float(depth_norm[mask1_near].mean())
    avg2_near = float(depth_norm[mask2_near].mean())

    # Relative difference: gap as a fraction of the closer object's depth.
    closer_depth = min(avg1_near, avg2_near)
    relative_diff = np.abs(avg1_near - avg2_near) / max(closer_depth, 1e-3)

    if relative_diff > depth_threshold:
        return np.zeros_like(mask1, dtype=np.uint8)
    return contact_zone.astype(np.uint8) * 255
