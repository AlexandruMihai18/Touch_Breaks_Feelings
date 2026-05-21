"""Depth Repair Reviewer
---------------------
EPIC Kitchen annotations sorted by touch coverage (largest first).
Four-panel A/B view:
  GT annotation | v1 (global avg) | v2 (pixel-level local) | depth map
"""

import json
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from inference_script.shared import EK_ANNO_DIR, EK_COVERAGE_PATH, OUT_H
from touch_detection_alg.depth import colorize_depth, predict_depth, sharpen_depth
from touch_detection_alg.touch import compute_touch_region_v2
from inference_script.src.deprecated.touch import compute_touch_region_with_depth as compute_touch_region_v1
from inference_script.src.visualization import draw_dual_mask_viz

# ---- state ----

_entries: list[dict] = []
_frame_idx = [0]
_cache: dict = {}   # img, obj, hand, gt_touch, depth, depth_refined (None until computed)

# ---- coverage helpers ----

def _load_global_coverage() -> dict[str, float]:
    lookup: dict[str, float] = {}
    if not EK_COVERAGE_PATH.exists():
        return lookup
    for line in EK_COVERAGE_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            lookup[" ".join(parts[5:])] = float(parts[0])
        except ValueError:
            continue
    return lookup


_global_coverage: dict[str, float] = _load_global_coverage()


def _entry_coverage(e: dict) -> float:
    path = e.get("touch_mask_path", "")
    p = Path(path) if path else None
    if p and p.exists():
        arr = np.array(Image.open(p).convert("L"))
        return float(np.count_nonzero(arr)) / arr.size if arr.size > 0 else 0.0
    return _global_coverage.get(e.get("object_name", ""), 0.0)


# ---- data loading ----

def _load_ek_entries(split: str) -> list[dict]:
    anno_file = EK_ANNO_DIR / f"{split}.json"
    if not anno_file.exists():
        return []
    result = []
    for e in json.loads(anno_file.read_text()):
        norm = dict(e)
        if "target_path" in norm and "touch_mask_path" not in norm:
            norm["touch_mask_path"] = norm.pop("target_path")
        result.append(norm)
    return result


def _load_and_sort(split: str) -> list[dict]:
    entries = _load_ek_entries(split)
    for e in entries:
        e["_coverage"] = _entry_coverage(e)
    return sorted(entries, key=lambda e: e["_coverage"], reverse=True)


# ---- frame rendering ----

_BLANK = np.zeros((OUT_H, OUT_H, 3), dtype=np.uint8)


def _load_frame(entry: dict) -> dict:
    """Load image and masks. depth is set to None — computed separately."""
    try:
        img      = np.array(Image.open(entry["image_path"]).convert("RGB"))
        obj      = np.array(Image.open(entry["object_mask_path"]).convert("L"))
        hand_key = "hand_mask_path" if "hand_mask_path" in entry else "stick_mask_path"
        hand     = np.array(Image.open(entry[hand_key]).convert("L"))
        gt_touch = np.array(Image.open(entry["touch_mask_path"]).convert("L"))
        return {"img": img, "obj": obj, "hand": hand, "gt_touch": gt_touch, "depth": None}
    except Exception as exc:
        print(f"  ⚠ Frame load failed: {exc}")
        return {}


def _ensure_depth(cache: dict) -> None:
    """Run depth prediction on the cached frame if not already done."""
    if not cache or cache.get("depth") is not None:
        return
    try:
        cache["depth"] = predict_depth(cache["img"])
    except Exception as exc:
        print(f"  ⚠ Depth prediction failed: {exc}")


def _ensure_refined_depth(cache: dict, radius: int, eps: float) -> None:
    """Compute guided-filter refined depth, caching by (radius, eps)."""
    if not cache or cache.get("depth") is None:
        return
    if cache.get("_refine_key") == (radius, eps) and cache.get("depth_refined") is not None:
        return
    try:
        cache["depth_refined"] = sharpen_depth(cache["depth"], cache["img"], radius=radius, eps=eps)
        cache["_refine_key"] = (radius, eps)
    except Exception as exc:
        print(f"  ⚠ Depth refinement failed: {exc}")


def _render(
    arrays: dict,
    dilation: int,
    v1_threshold: float,
    v1_window: float,
    v2_threshold: float,
    v2_local_radius: int,
    use_refine: bool,
    refine_radius: int,
    refine_eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (gt_panel, v1_panel, v2_panel, depth_panel)."""
    if not arrays:
        return _BLANK, _BLANK, _BLANK, _BLANK

    img      = arrays["img"]
    obj      = arrays["obj"]
    hand     = arrays["hand"]
    gt_touch = arrays["gt_touch"]
    depth    = arrays.get("depth")

    gt_panel = draw_dual_mask_viz(img, obj, hand, gt_touch, [], [])

    if depth is None:
        return gt_panel, _BLANK, _BLANK, _BLANK

    if use_refine:
        _ensure_refined_depth(arrays, refine_radius, refine_eps)
        refined = arrays.get("depth_refined")
        depth = refined if refined is not None else depth

    v1 = compute_touch_region_v1(
        hand, obj, depth,
        dilation_radius=dilation,
        depth_threshold=v1_threshold,
        window_factor=v1_window,
    )
    v2 = compute_touch_region_v2(
        hand, obj, depth,
        dilation_radius=dilation,
        abs_d_threshold=v2_threshold,
        local_radius=v2_local_radius,
    )

    return (
        gt_panel,
        draw_dual_mask_viz(img, obj, hand, v1, [], []),
        draw_dual_mask_viz(img, obj, hand, v2, [], []),
        colorize_depth(depth),
    )


def _frame_label() -> str:
    if not _entries:
        return "No frames loaded"
    e    = _entries[_frame_idx[0]]
    vid  = e.get("video_id", "?")
    obj  = e.get("object_name", "")
    stem = Path(e.get("image_path", "?")).stem
    cov  = e.get("_coverage", 0.0)
    n    = len(_entries)
    return f"{_frame_idx[0] + 1}/{n}  ·  {vid}  ·  {obj}  ·  {stem}  ·  cov={cov:.1%}"


# ---- callbacks ----

def _go_to(idx, dilation, v1_thr, v1_win, v2_thr, v2_lr, use_refine, r_radius, r_eps):
    _frame_idx[0] = max(0, min(idx, len(_entries) - 1))
    _cache.clear()
    if not _entries:
        return "No frames", _BLANK, _BLANK, _BLANK, _BLANK
    _cache.update(_load_frame(_entries[_frame_idx[0]]))
    _ensure_depth(_cache)
    gt, v1, v2, depth_viz = _render(_cache, dilation, v1_thr, v1_win, v2_thr, v2_lr, use_refine, r_radius, r_eps)
    return _frame_label(), gt, v1, v2, depth_viz


def go_prev(dilation, v1_thr, v1_win, v2_thr, v2_lr, use_refine, r_radius, r_eps):
    return _go_to(_frame_idx[0] - 1, dilation, v1_thr, v1_win, v2_thr, v2_lr, use_refine, r_radius, r_eps)


def go_next(dilation, v1_thr, v1_win, v2_thr, v2_lr, use_refine, r_radius, r_eps):
    return _go_to(_frame_idx[0] + 1, dilation, v1_thr, v1_win, v2_thr, v2_lr, use_refine, r_radius, r_eps)


def on_params_change(dilation, v1_thr, v1_win, v2_thr, v2_lr, use_refine, r_radius, r_eps):
    if not _cache:
        return _BLANK, _BLANK, _BLANK
    _ensure_depth(_cache)
    _, v1, v2, depth_viz = _render(_cache, dilation, v1_thr, v1_win, v2_thr, v2_lr, use_refine, r_radius, r_eps)
    return v1, v2, depth_viz


def on_split_change(split, dilation, v1_thr, v1_win, v2_thr, v2_lr, use_refine, r_radius, r_eps):
    _entries.clear()
    _cache.clear()
    _frame_idx[0] = 0
    _entries.extend(_load_and_sort(split))
    if not _entries:
        return "No frames", _BLANK, _BLANK, _BLANK, _BLANK
    _cache.update(_load_frame(_entries[0]))
    _ensure_depth(_cache)
    gt, v1, v2, depth_viz = _render(_cache, dilation, v1_thr, v1_win, v2_thr, v2_lr, use_refine, r_radius, r_eps)
    return _frame_label(), gt, v1, v2, depth_viz


# ---- init ----
# Depth is NOT computed here to avoid loading the model at app startup.

_entries.extend(_load_and_sort("train"))
if _entries:
    _cache.update(_load_frame(_entries[0]))
    _init_gt, _init_v1, _init_v2, _init_depth = _render(_cache, 10, 0.3, 2.0, 0.05, 10, True, 4, 0.1)
    _init_label = _frame_label()
else:
    _init_gt = _init_v1 = _init_v2 = _init_depth = _BLANK
    _init_label = "No frames loaded"


# ---- UI ----

with gr.Blocks() as tab:
    gr.Markdown(
        "## Depth Repair — A/B Comparison\n"
        "EPIC Kitchen annotations sorted by touch coverage (largest first).\n\n"
        "| Panel | What it shows |\n"
        "|---|---|\n"
        "| **GT** | Ground-truth annotation (original touch mask) |\n"
        "| **v1** | Global avg depths, relative threshold (whole-zone keep/reject) |\n"
        "| **v2** | Pixel-level local depth gap — each contact pixel judged independently |\n"
        "| **Depth** | Depth map (inferno colormap) |\n\n"
        "> Depth is computed on first navigation — v1/v2/depth panels are blank until then."
    )

    with gr.Row():
        ek_split_radio = gr.Radio(
            choices=["train", "val"], value="train",
            label="Split", interactive=True,
        )

    frame_label = gr.Textbox(value=_init_label, label="Frame", interactive=False)

    with gr.Row():
        prev_btn = gr.Button("← Prev", variant="secondary")
        next_btn = gr.Button("Next →", variant="secondary")

    gr.Markdown("### Shared parameter")
    dilation_slider = gr.Slider(
        minimum=1, maximum=60, step=1, value=10,
        label="Dilation radius (px)",
        info="Contact zone half-width — applies to both methods.",
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("#### v1 — global avg (whole-zone decision)")
            v1_threshold_slider = gr.Slider(
                minimum=0.01, maximum=1.0, step=0.01, value=0.3,
                label="Relative depth threshold",
                info="Max relative gap between whole-mask avg depths. Lower = stricter.",
            )
            v1_window_slider = gr.Slider(
                minimum=1.0, maximum=5.0, step=0.1, value=2.0,
                label="Window factor",
                info="Near-face sampling radius multiplier (depth-adaptive).",
            )
        with gr.Column():
            gr.Markdown("#### v2 — pixel-level local depth gap")
            v2_threshold_slider = gr.Slider(
                minimum=0.01, maximum=0.50, step=0.01, value=0.05,
                label="Abs. depth gap threshold",
                info="Per-pixel threshold: keep contact pixel p if "
                     "|hand_depth(p) − obj_depth(p)| ≤ this. 0.05 = strict, 0.20 = lenient.",
            )
            v2_local_radius_slider = gr.Slider(
                minimum=2, maximum=40, step=1, value=10,
                label="Local window radius (px)",
                info="Half-width of the box window used to estimate hand/object depth at each pixel. "
                     "Smaller = more spatial detail but noisier; larger = smoother but less localised. "
                     "Default = dilation radius.",
            )

    gr.Markdown("### Depth post-processing (guided image filter)")
    with gr.Row():
        refine_cb = gr.Checkbox(
            label="Sharpen depth with guided filter",
            value=True,
            info="Uses RGB image as edge guide (He et al. 2010) to snap blurry depth boundaries. "
                 "Cached per frame — only recomputes when radius/ε change.",
        )
    with gr.Row():
        refine_radius_slider = gr.Slider(
            minimum=1, maximum=32, step=1, value=4,
            label="Guided-filter radius (px)",
        )
        refine_eps_slider = gr.Slider(
            minimum=0.001, maximum=0.2, step=0.001, value=0.1,
            label="Regularisation ε",
        )

    with gr.Row():
        img_gt    = gr.Image(value=_init_gt,    label="GT annotation",    height=OUT_H, interactive=False)
        img_v1    = gr.Image(value=_init_v1,    label="v1 – global avg",  height=OUT_H, interactive=False)
        img_v2    = gr.Image(value=_init_v2,    label="v2 – pixel-level", height=OUT_H, interactive=False)
        img_depth = gr.Image(value=_init_depth, label="Depth map",        height=OUT_H, interactive=False)

    _touch_params  = [dilation_slider, v1_threshold_slider, v1_window_slider,
                      v2_threshold_slider, v2_local_radius_slider]
    _refine_params = [refine_cb, refine_radius_slider, refine_eps_slider]
    _params        = _touch_params + _refine_params
    _nav_outputs   = [frame_label, img_gt, img_v1, img_v2, img_depth]

    prev_btn.click(fn=go_prev, inputs=_params, outputs=_nav_outputs)
    next_btn.click(fn=go_next, inputs=_params, outputs=_nav_outputs)

    ek_split_radio.change(
        fn=on_split_change,
        inputs=[ek_split_radio] + _params,
        outputs=_nav_outputs,
    )

    for ctrl in _params:
        ctrl.change(
            fn=on_params_change,
            inputs=_params,
            outputs=[img_v1, img_v2, img_depth],
        )
