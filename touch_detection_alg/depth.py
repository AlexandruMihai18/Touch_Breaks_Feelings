import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter

_processor = None
_model = None

# ── Blur / sharpening knobs ───────────────────────────────────────────────────
# Run diagnose_blur_scores() on a sample of your frames first, then set:
#   _BLUR_SHARP_THRESHOLD  → p75 of the distribution (frames above this are sharp)
#   _BLUR_HEAVY_THRESHOLD  → p25 (frames below this get maximum sharpening)
# Unsharp mask: percent controls strength (50 = mild, 300 = very aggressive)
_BLUR_SHARP_THRESHOLD = 500.0   # above → skip sharpening
_BLUR_HEAVY_THRESHOLD = 100.0   # below → max sharpening
_SHARP_PERCENT_MAX = 250        # strength for very blurry frames
_SHARP_PERCENT_MIN = 60         # strength for mildly blurry frames
# ─────────────────────────────────────────────────────────────────────────────


def _load_model():
    global _processor, _model
    if _model is None:
        from transformers import (AutoImageProcessor,
                                  AutoModelForDepthEstimation)

        _processor = AutoImageProcessor.from_pretrained(
            "depth-anything/Depth-Anything-V2-Small-hf"
        )
        _model = AutoModelForDepthEstimation.from_pretrained(
            "depth-anything/Depth-Anything-V2-Small-hf"
        )
        _model.eval()
        if torch.cuda.is_available():
            _model.cuda()


def _blur_score(frame_rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _apply_clahe(pil_image: Image.Image, clip_limit: float = 2.0, tile_size: int = 8) -> Image.Image:
    """CLAHE on the L channel of LAB colorspace — boosts local contrast patch-by-patch.

    Particularly effective for motion-blurred frames: restores edge contrast that
    unsharp-mask then amplifies, giving the depth model much cleaner input.
    """
    img = np.array(pil_image)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def _adaptive_sharpen(pil_image: Image.Image, score: float) -> Image.Image:
    if score >= _BLUR_SHARP_THRESHOLD:
        return pil_image
    if score <= _BLUR_HEAVY_THRESHOLD:
        t = 0.0
    else:
        t = (score - _BLUR_HEAVY_THRESHOLD) / (_BLUR_SHARP_THRESHOLD - _BLUR_HEAVY_THRESHOLD)
    percent = int(_SHARP_PERCENT_MAX - t * (_SHARP_PERCENT_MAX - _SHARP_PERCENT_MIN))
    # heavier blur → larger radius to capture wider edge spread
    radius = 3 if score < _BLUR_HEAVY_THRESHOLD else 2
    return pil_image.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=3))


def sharpen_image(image_np: np.ndarray) -> Image.Image:
    """Return a sharpened PIL image using the current blur thresholds.

    Uses the same adaptive unsharp-mask pipeline as depth prediction so that
    Grounding DINO receives a consistently pre-processed input.
    """
    pil = Image.fromarray(image_np).convert("RGB")
    return _adaptive_sharpen(pil, _blur_score(image_np))


def calibrate_blur_thresholds(frames: list) -> tuple[float, float]:
    """Set blur thresholds from the actual score distribution of the given frames.

    Computes Laplacian-variance scores for every frame, then assigns:
      _BLUR_HEAVY_THRESHOLD = p25  (below → max sharpening)
      _BLUR_SHARP_THRESHOLD = p75  (above → no sharpening)

    Returns (heavy_threshold, sharp_threshold).
    """
    global _BLUR_HEAVY_THRESHOLD, _BLUR_SHARP_THRESHOLD
    scores = np.array([_blur_score(f) for f in frames])
    _BLUR_HEAVY_THRESHOLD = float(np.percentile(scores, 25))
    _BLUR_SHARP_THRESHOLD = float(np.percentile(scores, 75))
    return _BLUR_HEAVY_THRESHOLD, _BLUR_SHARP_THRESHOLD


def predict_depth(image: np.ndarray, clahe_clip: float = 0.0) -> np.ndarray:
    """Run Depth-Anything V2 on an RGB numpy image.

    Returns a float32 array of shape (H, W) with values in [0, 1],
    normalized relative to the scene's min/max depth.

    clahe_clip > 0 applies CLAHE contrast enhancement before unsharp sharpening,
    which helps with motion-blurred frames by restoring local edge contrast first.
    """
    _load_model()
    pil_image = Image.fromarray(image).convert("RGB")
    if clahe_clip > 0.0:
        pil_image = _apply_clahe(pil_image, clip_limit=clahe_clip)
    pil_image = _adaptive_sharpen(pil_image, _blur_score(np.array(pil_image)))
    device = next(_model.parameters()).device
    inputs = _processor(images=[pil_image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _model(**inputs)
        predicted_depth = outputs.predicted_depth

    depth = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=pil_image.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    depth_np = depth.cpu().float().numpy()
    d_min, d_max = depth_np.min(), depth_np.max()
    if d_max > d_min:
        return (depth_np - d_min) / (d_max - d_min)
    return np.zeros_like(depth_np)


def colorize_depth(depth_normalized: np.ndarray) -> np.ndarray:
    """Convert a [0, 1] depth map to an RGB uint8 image (inferno colormap)."""
    import matplotlib.cm as cm

    colormap = cm.get_cmap("inferno")
    return (colormap(depth_normalized)[:, :, :3] * 255).astype(np.uint8)


def guided_filter(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int = 8,
    eps: float = 0.01,
) -> np.ndarray:
    """Guided image filter (He et al. 2010) — pure numpy/scipy, no extra deps.

    Uses the grayscale guide image to transfer sharp edges from the RGB photo
    onto the blurry depth map.  Works because depth boundaries co-occur with
    intensity edges in the RGB image.

    Parameters
    ----------
    guide : float32 array (H, W), range [0, 1]  — typically grayscale RGB
    src   : float32 array (H, W), range [0, 1]  — depth map to sharpen
    radius: box-filter half-width in pixels
    eps   : regularisation (larger → smoother, ignores weak edges)
    """
    from scipy.ndimage import uniform_filter

    w = 2 * radius + 1

    mean_I  = uniform_filter(guide, w)
    mean_p  = uniform_filter(src,   w)
    mean_Ip = uniform_filter(guide * src, w)
    mean_II = uniform_filter(guide * guide, w)

    cov_Ip = mean_Ip - mean_I * mean_p
    var_I  = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = uniform_filter(a, w)
    mean_b = uniform_filter(b, w)

    return np.clip(mean_a * guide + mean_b, 0.0, 1.0).astype(np.float32)


def sharpen_depth(
    depth: np.ndarray,
    image_rgb: np.ndarray,
    radius: int = 8,
    eps: float = 0.01,
) -> np.ndarray:
    """Refine a monocular depth map using the RGB image as an edge guide.

    Converts the RGB image to grayscale and runs a guided image filter so that
    sharp colour/intensity boundaries in the photo propagate into the depth map.
    This improves depth accuracy at hand-object interaction boundaries without
    requiring any additional packages.

    Parameters
    ----------
    depth     : float32 (H, W) in [0, 1] — raw depth from predict_depth
    image_rgb : uint8  (H, W, 3) — original RGB frame
    radius    : guided-filter box radius (pixels)
    eps       : regularisation strength
    """
    guide = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    depth_f = depth.astype(np.float32)
    return guided_filter(guide, depth_f, radius=radius, eps=eps)
