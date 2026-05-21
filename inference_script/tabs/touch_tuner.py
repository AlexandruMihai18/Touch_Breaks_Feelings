"""
Touch Dilation Tuner
--------------------
Load hand + object masks from any dataset and interactively adjust the
dilation radius used in compute_touch_region.  Compare the VISOR ground-truth
touch mask (often a 1-2px boundary line) against the dilated version side-by-side
to find the right dilation radius before baking it into the pipeline.
"""
import json
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from inference_script.shared import (
    DATASET_NAMES, DATASETS, EK_ANNO_DIR, GH_MASKS_ROOT, MASKS_ROOT, OUT_H,
)
from touch_detection_alg.touch import compute_touch_region
from inference_script.src.visualization import draw_dual_mask_viz

# ---- state ----

_entries: list[dict] = []   # flat list of all frame dicts for current dataset
_frame_idx = [0]
_dataset_name = [DATASET_NAMES[0]]

# Cached arrays for the current frame — reused on slider change to avoid re-reading disk
_cache: dict = {}   # keys: img, hand, obj, gt_touch


# ---- data loading ----

def _collect_manifest_entries(masks_root: Path) -> list[dict]:
    entries = []
    if not masks_root.exists():
        return entries
    for folder in sorted(d for d in masks_root.iterdir() if d.is_dir()):
        manifest = folder / "manifest.json"
        if not manifest.exists():
            continue
        data = json.loads(manifest.read_text())
        entries.extend(data.get("frames", []))
    return entries


def _collect_ek_entries(split: str) -> list[dict]:
    anno_file = EK_ANNO_DIR / f"{split}.json"
    if not anno_file.exists():
        return []
    entries = []
    for e in json.loads(anno_file.read_text()):
        norm = dict(e)
        if "target_path" in norm and "touch_mask_path" not in norm:
            norm["touch_mask_path"] = norm.pop("target_path")
        entries.append(norm)
    return entries


def _load_dataset(dataset_name: str, ek_split: str = "train") -> list[dict]:
    if dataset_name == "EPIC Kitchen":
        return _collect_ek_entries(ek_split)
    _, masks_root = DATASETS[dataset_name]
    return _collect_manifest_entries(masks_root)


# ---- rendering ----

def _load_frame_arrays(entry: dict) -> dict:
    """Read masks from disk and return numpy arrays, or empty arrays on error."""
    try:
        img     = np.array(Image.open(entry["image_path"]).convert("RGB"))
        h, w    = img.shape[:2]
        obj     = np.array(Image.open(entry["object_mask_path"]).convert("L"))
        hand_key = "hand_mask_path" if "hand_mask_path" in entry else "stick_mask_path"
        hand    = np.array(Image.open(entry[hand_key]).convert("L"))
        gt_touch = np.array(Image.open(entry["touch_mask_path"]).convert("L"))
        return {"img": img, "hand": hand, "obj": obj, "gt_touch": gt_touch}
    except Exception as exc:
        print(f"  ⚠ Could not load frame: {exc}")
        blank = np.zeros((OUT_H, OUT_H, 3), dtype=np.uint8)
        empty = np.zeros((OUT_H, OUT_H), dtype=np.uint8)
        return {"img": blank, "hand": empty, "obj": empty, "gt_touch": empty}


def _render_panels(arrays: dict, dilation: int):
    """Return (masks_panel, gt_touch_panel, computed_panel)."""
    img      = arrays["img"]
    hand     = arrays["hand"]
    obj      = arrays["obj"]
    gt_touch = arrays["gt_touch"]
    zero     = np.zeros_like(gt_touch)

    computed = compute_touch_region(hand, obj, dilation_radius=dilation)

    masks_panel    = draw_dual_mask_viz(img, obj, hand, zero,     [], [])
    gt_panel       = draw_dual_mask_viz(img, obj, hand, gt_touch, [], [])
    computed_panel = draw_dual_mask_viz(img, obj, hand, computed, [], [])
    return masks_panel, gt_panel, computed_panel


def _frame_label() -> str:
    if not _entries:
        return "No frames loaded"
    e   = _entries[_frame_idx[0]]
    vid = e.get("video_id") or Path(e["image_path"]).parent.name
    return f"Frame {_frame_idx[0] + 1}/{len(_entries)}  ·  {vid}  ·  {Path(e['image_path']).name}  [{e.get('type', '')}]"


# ---- callbacks ----

def _go_to(idx: int, dilation: int):
    _frame_idx[0] = max(0, min(idx, len(_entries) - 1))
    _cache.clear()
    if not _entries:
        blank = np.zeros((OUT_H, OUT_H, 3), dtype=np.uint8)
        return "No frames", blank, blank, blank
    _cache.update(_load_frame_arrays(_entries[_frame_idx[0]]))
    masks, gt, comp = _render_panels(_cache, dilation)
    return _frame_label(), masks, gt, comp


def go_next(dilation: int):
    return _go_to(_frame_idx[0] + 1, dilation)


def go_prev(dilation: int):
    return _go_to(_frame_idx[0] - 1, dilation)


def on_dilation_change(dilation: int):
    if not _cache:
        blank = np.zeros((OUT_H, OUT_H, 3), dtype=np.uint8)
        return blank
    _, _, comp = _render_panels(_cache, dilation)
    return comp


def on_dataset_change(dataset_name: str, ek_split: str):
    _dataset_name[0] = dataset_name
    _entries.clear()
    _cache.clear()
    _frame_idx[0] = 0
    is_ek = dataset_name == "EPIC Kitchen"
    _entries.extend(_load_dataset(dataset_name, ek_split))
    split_update = gr.update(visible=is_ek)
    if not _entries:
        blank = np.zeros((OUT_H, OUT_H, 3), dtype=np.uint8)
        return "No frames loaded", split_update, blank, blank, blank
    _cache.update(_load_frame_arrays(_entries[0]))
    masks, gt, comp = _render_panels(_cache, 10)
    return _frame_label(), split_update, masks, gt, comp


def on_split_change(ek_split: str, dilation: int):
    _entries.clear()
    _cache.clear()
    _frame_idx[0] = 0
    _entries.extend(_collect_ek_entries(ek_split))
    if not _entries:
        blank = np.zeros((OUT_H, OUT_H, 3), dtype=np.uint8)
        return "No frames loaded", blank, blank, blank
    _cache.update(_load_frame_arrays(_entries[0]))
    masks, gt, comp = _render_panels(_cache, dilation)
    return _frame_label(), masks, gt, comp


# ---- init ----

_entries.extend(_load_dataset(DATASET_NAMES[0]))
if _entries:
    _cache.update(_load_frame_arrays(_entries[0]))
    _init_masks, _init_gt, _init_comp = _render_panels(_cache, 10)
    _init_label = _frame_label()
else:
    _blank = np.zeros((OUT_H, OUT_H, 3), dtype=np.uint8)
    _init_masks = _init_gt = _init_comp = _blank
    _init_label = "No frames loaded"


# ---- UI ----

with gr.Blocks() as tab:
    gr.Markdown(
        "## Touch Dilation Tuner\n"
        "Adjust **dilation radius** to find the right contact-zone thickness. "
        "**Left**: hand (orange) + object (blue) — no touch.  "
        "**Middle**: VISOR ground-truth touch (magenta).  "
        "**Right**: computed touch at current dilation (magenta)."
    )

    with gr.Row():
        dataset_radio = gr.Radio(
            choices=DATASET_NAMES, value=DATASET_NAMES[0],
            label="Dataset", interactive=True,
        )
        ek_split_radio = gr.Radio(
            choices=["train", "val"], value="train",
            label="Split", visible=False, interactive=True,
        )

    frame_label = gr.Textbox(value=_init_label, label="Frame", interactive=False)

    with gr.Row():
        prev_btn = gr.Button("← Prev", variant="secondary")
        next_btn = gr.Button("Next →", variant="secondary")

    dilation_slider = gr.Slider(
        minimum=1, maximum=60, step=1, value=10,
        label="Dilation radius (px)",
        info="Radius used in compute_touch_region. Slides update the right panel live.",
    )

    with gr.Row():
        img_masks    = gr.Image(value=_init_masks,   label="Hand + Object (no touch)", height=OUT_H, interactive=False)
        img_gt       = gr.Image(value=_init_gt,      label="Ground-truth touch",        height=OUT_H, interactive=False)
        img_computed = gr.Image(value=_init_comp,    label="Computed touch (dilated)",   height=OUT_H, interactive=False)

    # output lists
    _nav_outputs    = [frame_label, img_masks, img_gt, img_computed]
    _switch_outputs = [frame_label, ek_split_radio, img_masks, img_gt, img_computed]
    _split_outputs  = [frame_label, img_masks, img_gt, img_computed]

    dataset_radio.change(
        fn=on_dataset_change,
        inputs=[dataset_radio, ek_split_radio],
        outputs=_switch_outputs,
    )
    ek_split_radio.change(
        fn=on_split_change,
        inputs=[ek_split_radio, dilation_slider],
        outputs=_split_outputs,
    )
    prev_btn.click(fn=go_prev, inputs=[dilation_slider], outputs=_nav_outputs)
    next_btn.click(fn=go_next, inputs=[dilation_slider], outputs=_nav_outputs)
    dilation_slider.change(
        fn=on_dilation_change,
        inputs=[dilation_slider],
        outputs=[img_computed],
    )
