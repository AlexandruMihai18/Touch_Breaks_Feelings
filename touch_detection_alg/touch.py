import cv2
import numpy as np


def _dilate_mask(mask, radius):
    """Binary dilation — cv2 ellipse fast path, scipy fallback."""
    try:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        return cv2.dilate((mask > 0).astype(np.uint8), kernel).astype(bool)
    except Exception:
        pass
    try:
        from scipy.ndimage import binary_dilation
        struct = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
        return binary_dilation(mask > 0, structure=struct)
    except ImportError:
        result = mask > 0
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    result = result | np.roll(np.roll(mask > 0, dy, 0), dx, 1)
        return result


def _erode_mask(mask, radius):
    """Binary erosion — cv2 ellipse fast path, scipy fallback."""
    try:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        return cv2.erode((mask > 0).astype(np.uint8), kernel).astype(bool)
    except Exception:
        pass
    try:
        from scipy.ndimage import binary_erosion
        struct = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
        return binary_erosion(mask > 0, structure=struct)
    except ImportError:
        result = mask > 0
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    result = result & np.roll(np.roll(mask > 0, dy, 0), dx, 1)
        return result


def _local_mean_depth(depth: np.ndarray, mask_bin: np.ndarray, radius: int) -> np.ndarray:
    """Per-pixel mean depth of `mask_bin` pixels inside a (2r+1)×(2r+1) box window.

    Returns NaN where no mask pixel falls in the window.
    Uses cv2.boxFilter (O(N) regardless of radius) for speed.
    """
    ksize = (2 * radius + 1, 2 * radius + 1)
    depth_sum = cv2.boxFilter(depth * mask_bin, ddepth=-1, ksize=ksize, normalize=False)
    count     = cv2.boxFilter(mask_bin,         ddepth=-1, ksize=ksize, normalize=False)
    return np.where(count > 0, depth_sum / np.maximum(count, 1e-9), np.nan)


def compute_touch_region(mask1, mask2, dilation_radius=10):
    """Find the contact zone between two binary masks.

    Both masks are dilated by dilation_radius pixels; their intersection is
    the region that straddles the boundary between the two objects.
    Returns an empty mask when the objects are too far apart to interact.
    """
    d1 = _dilate_mask(mask1, dilation_radius)
    d2 = _dilate_mask(mask2, dilation_radius)
    return (d1 & d2).astype(np.uint8) * 255


def compute_touch_region_v2(
    mask1,
    mask2,
    depth_map,
    erode_radius: int = 0,
    dilation_radius: int = 10,
    abs_d_threshold: float = 0.05,
    local_radius: int | None = 10,
):
    """Pixel-level contact zone filtered by local depth gap.

    Improvement over the global-median approach: instead of one keep/reject
    decision for the whole zone, every pixel in the contact zone gets its own
    local depth estimate for each mask and is kept or discarded individually.
    This correctly handles mixed zones — e.g. a hand touching one corner of a
    large object while hovering in front of another corner.

    For each pixel p in the contact zone:
        hand_depth(p) = mean depth of hand pixels within local_radius of p
        obj_depth(p)  = mean depth of object pixels within local_radius of p
        keep p  iff  |hand_depth(p) − obj_depth(p)| ≤ abs_d_threshold

    Because the contact zone is the intersection of the two dilated masks,
    every pixel in it is within dilation_radius of at least one hand pixel
    AND one object pixel.  Setting local_radius = dilation_radius (the default)
    therefore guarantees that both estimates are always defined.

    Parameters
    ----------
    mask1           : hand mask (uint8, 0/255)
    mask2           : object mask (uint8, 0/255)
    depth_map       : depth array (arbitrary scale, normalised internally to [0,1])
    dilation_radius : contact zone half-width in pixels
    abs_d_threshold : max absolute depth gap [0,1] to keep a pixel.
                      0.05 = strict, 0.20 = lenient.
    local_radius    : half-width of the window used to estimate local depth.
                      Defaults to dilation_radius.  Smaller → more spatial detail
                      but noisier estimates; larger → smoother but less localised.
    """
    if erode_radius > 0:
        mask1 = _erode_mask(mask1, erode_radius)
        mask2 = _erode_mask(mask2, erode_radius)

    d1 = _dilate_mask(mask1, dilation_radius)
    d2 = _dilate_mask(mask2, dilation_radius)
    contact_zone = d1 & d2

    if not contact_zone.any():
        return np.zeros_like(mask1, dtype=np.uint8)
    if not (mask1 > 0).any() or not (mask2 > 0).any():
        return contact_zone.astype(np.uint8) * 255

    depth_norm = depth_map.astype(np.float32)
    d_min, d_max = depth_norm.min(), depth_norm.max()
    if d_max > d_min:
        depth_norm = (depth_norm - d_min) / (d_max - d_min)

    r = dilation_radius if local_radius is None else local_radius
    hand_bin = (mask1 > 0).astype(np.float32)
    obj_bin  = (mask2 > 0).astype(np.float32)

    hand_local = _local_mean_depth(depth_norm, hand_bin, r)
    obj_local  = _local_mean_depth(depth_norm, obj_bin,  r)

    depth_gap = np.abs(hand_local - obj_local)

    # Keep only contact zone pixels with a valid, small local depth gap.
    # np.isfinite guards the rare edge case where the window contains no mask pixel.
    valid = contact_zone & np.isfinite(depth_gap) & (depth_gap <= abs_d_threshold)

    if not valid.any():
        return np.zeros_like(mask1, dtype=np.uint8)

    # Recovery: filter valid pixels by connected-component size.
    # Components whose equivalent radius (sqrt(area/π)) >= r/2 are large enough
    # to trust; dilate them back into the contact zone. Smaller components are
    # too thin/sparse and are discarded.
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        valid.astype(np.uint8), connectivity=8
    )
    min_equiv_radius = r * 6 / 10.0
    filtered = np.zeros(valid.shape, dtype=bool)
    for lbl in range(1, n_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if (area / np.pi) ** 0.5 >= min_equiv_radius:
            filtered |= labels == lbl

    if not filtered.any():
        return np.zeros_like(mask1, dtype=np.uint8)

    recovered = _dilate_mask(filtered, r) & contact_zone
    return recovered.astype(np.uint8) * 255
