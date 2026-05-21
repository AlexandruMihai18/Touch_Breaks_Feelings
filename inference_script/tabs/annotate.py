"""Manual annotation tab — scribble-driven SAM + SegGPT propagation."""

import json
import random
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from inference_script.shared import FRAMES_ROOT, MASKS_ROOT, OUT_H
from touch_detection_alg.depth import (calibrate_blur_thresholds,
                                         colorize_depth, predict_depth)
from inference_script.src.sam import run_sam
from touch_detection_alg.greatest_hits.detection import run_sam_text
from inference_script.src.seggpt import run_seggpt
from touch_detection_alg.touch import compute_touch_region_v2
from inference_script.src.visualization import draw_dual_mask_viz, draw_stick_viz

# ---- state ----

_folders: list[Path] = (
    sorted(d for d in FRAMES_ROOT.iterdir() if d.is_dir())
    if FRAMES_ROOT.exists()
    else []
)
_folder_idx = [0]
_frames: list[np.ndarray] = []
_frame_paths: list[Path] = []
_masks: dict = {}
_prompt_idx = [0]


# ---- helpers ----


def _load_folder(idx: int) -> tuple[str, list, list]:
    folder = _folders[idx]
    paths = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
    frames = [np.array(Image.open(p).convert("RGB")) for p in paths]
    return folder.name, frames, paths


def _pick_prompt() -> tuple[np.ndarray | None, str]:
    if not _frames:
        return None, "No frames loaded"
    _prompt_idx[0] = random.randint(0, len(_frames) - 1)
    stem = _frame_paths[_prompt_idx[0]].stem if _frame_paths else "unknown"
    return (
        np.array(_frames[_prompt_idx[0]]),
        f"Prompt: {stem} ({_prompt_idx[0] + 1}/{len(_frames)})",
    )


def _switch_folder(idx: int):
    _folder_idx[0] = idx
    name, frames, paths = _load_folder(idx)
    _frames.clear()
    _frames.extend(frames)
    _frame_paths.clear()
    _frame_paths.extend(paths)
    _masks.clear()
    heavy, sharp = calibrate_blur_thresholds(frames)
    frame, prompt_status = _pick_prompt()
    status = (
        f"Folder {idx + 1}/{len(_folders)}: {name} ({len(frames)} frames) — "
        f"{prompt_status} | blur thresholds auto-set: heavy={heavy:.0f} sharp={sharp:.0f}"
    )
    return name, frame, status, "", None, None


# ---- callbacks ----


def reload():
    frame, status = _pick_prompt()
    return frame, status, None, None, None


def identify_object(prompt_text):
    if not _frames:
        return None, "No frame loaded"
    prompt_text = prompt_text.strip()
    if not prompt_text:
        return None, "Enter an object prompt first"
    if not prompt_text.endswith("."):
        prompt_text += "."
    frame = _frames[_prompt_idx[0]]
    mask, box, score, sharpened = run_sam_text(frame, prompt_text)
    stem = _frame_paths[_prompt_idx[0]].stem if _frame_paths else "?"
    viz = draw_stick_viz(sharpened, mask, box, color=(30, 120, 255))
    if score == 0.0:
        return viz, f"No object detected for '{prompt_text}' in '{stem}'"
    return viz, f"Object detected in '{stem}' — prompt: '{prompt_text}' | score: {score:.3f} | box: {[round(v, 1) for v in box]}"


def process_prompt(editor, dilation_radius=10, abs_d_threshold=0.05, local_radius=10):
    if editor is None or editor.get("background") is None:
        return None, None, "Draw scribbles on the prompt frame first"
    bg = editor["background"]
    layers = editor.get("layers", [])
    blank = np.zeros((*bg.shape[:2], 4), dtype=np.uint8)
    s1 = layers[0] if len(layers) > 0 else blank
    s2 = layers[1] if len(layers) > 1 else blank

    mask1, pts1 = run_sam(bg, s1)
    mask2, pts2 = run_sam(bg, s2)

    depth_map = predict_depth(bg)
    touch = compute_touch_region_v2(
        mask1, mask2, depth_map, erode_radius=0, dilation_radius=dilation_radius, abs_d_threshold=abs_d_threshold, local_radius=local_radius
    )
    _masks.update(object_mask=mask1, hand_mask=mask2, prompt_img=bg, depth_map=depth_map)

    return (
        draw_dual_mask_viz(bg, mask1, mask2, touch, pts1, pts2),
        colorize_depth(depth_map),
        "Prompt processed — click 'Propagate & Save' to annotate all frames",
    )


def propagate_and_save(dilation_radius=10, abs_d_threshold=0.05, local_radius=10):
    if not _masks or not _frames:
        return "Process the prompt frame first"
    if not _folders:
        return "No folders loaded"

    folder_name = _folders[_folder_idx[0]].name
    out_dir = MASKS_ROOT / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_pil = Image.fromarray(_masks["prompt_img"]).convert("RGB")
    obj_pil = Image.fromarray(_masks["object_mask"]).convert("L")
    hand_pil = Image.fromarray(_masks["hand_mask"]).convert("L")

    annotations = {"video_id": folder_name, "frames": []}
    dataset_rows = []

    for idx, (frame, fp) in enumerate(zip(_frames, _frame_paths)):
        if idx == _prompt_idx[0]:
            obj_mask = _masks["object_mask"]
            hand_mask = _masks["hand_mask"]
            depth_map = _masks["depth_map"]
        else:
            frame_pil = Image.fromarray(frame).convert("RGB")
            obj_mask = run_seggpt(frame_pil, prompt_pil, obj_pil)
            hand_mask = run_seggpt(frame_pil, prompt_pil, hand_pil)
            depth_map = predict_depth(frame)

        touch = compute_touch_region_v2(
            obj_mask, hand_mask, depth_map, erode_radius=0, dilation_radius=int(dilation_radius), abs_d_threshold=float(abs_d_threshold), local_radius=int(local_radius)
        )
        stem = fp.stem

        touch_p = out_dir / f"{stem}_touch.png"
        obj_p   = out_dir / f"{stem}_object.png"
        hand_p  = out_dir / f"{stem}_hand.png"
        depth_p = out_dir / f"{stem}_depth.png"
        img_p   = out_dir / f"{stem}.jpg"

        Image.fromarray((touch > 0).astype(np.uint8) * 255, mode="L").save(touch_p)
        Image.fromarray((obj_mask > 0).astype(np.uint8) * 255, mode="L").save(obj_p)
        Image.fromarray((hand_mask > 0).astype(np.uint8) * 255, mode="L").save(hand_p)
        Image.fromarray(colorize_depth(depth_map)).save(depth_p)
        Image.fromarray(frame).save(img_p, quality=95)

        annotations["frames"].append({
            "image_path": str(img_p),
            "touch_mask_path": str(touch_p),
            "object_mask_path": str(obj_p),
            "hand_mask_path": str(hand_p),
            "depth_map_path": str(depth_p),
            "frame_index": idx,
        })
        dataset_rows.append({
            "image_path": str(img_p),
            "target_path": str(touch_p),
            "type": "touch" if np.any(touch > 0) else "no-touch",
            "video_id": folder_name,
        })

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(annotations, f, indent=2)
    with open(out_dir / "dataset.json", "w") as f:
        json.dump(dataset_rows, f, indent=2)

    return f"Saved {len(_frames)} frames to {out_dir}"


def go_next():
    if not _folders:
        return "", None, "No folders found", "", None, None
    return _switch_folder(min(_folder_idx[0] + 1, len(_folders) - 1))


def go_prev():
    if not _folders:
        return "", None, "No folders found", "", None, None
    return _switch_folder(max(_folder_idx[0] - 1, 0))


# ---- initialize ----

if _folders:
    _init_name, _init_frames, _init_paths = _load_folder(0)
    _frames.extend(_init_frames)
    _frame_paths.extend(_init_paths)
    _init_heavy, _init_sharp = calibrate_blur_thresholds(_init_frames)
    _init_frame, _init_prompt_status = _pick_prompt()
    _init_status = (
        f"Folder 1/{len(_folders)}: {_init_name}"
        f" ({len(_frames)} frames) — {_init_prompt_status}"
        f" | blur thresholds auto-set: heavy={_init_heavy:.0f} sharp={_init_sharp:.0f}"
    )
else:
    _init_name = "No folders found"
    _init_frame = None
    _init_status = "No frame folders found in data/manual_annotations/frames"


# ---- UI ----

with gr.Blocks() as tab:
    gr.Markdown(
        "## Batch Annotation\n"
        "A random frame is pre-selected as the prompt. "
        "Draw **object** scribbles on layer 1 and **hand** scribbles on layer 2, "
        "then propagate masks to all frames and save."
    )
    with gr.Row():
        with gr.Column(scale=1):
            folder_display = gr.Textbox(
                label="Current Folder", value=_init_name, interactive=False
            )
            status_box = gr.Textbox(
                label="Status", value=_init_status, interactive=False
            )
            dilation = gr.Slider(
                minimum=1, maximum=60, step=1, value=10,
                label="Dilation radius (px)",
                info="Contact zone half-width — applies to both methods.",
            )
            depth_threshold = gr.Slider(
                minimum=0.01, maximum=0.50, step=0.01, value=0.05,
                label="Abs. depth gap threshold",
                info="Per-pixel threshold: keep contact pixel p if "
                     "|hand_depth(p) − obj_depth(p)| ≤ this. 0.05 = strict, 0.20 = lenient.",
            )
            local_radius_slider = gr.Slider(
                minimum=2, maximum=40, step=1, value=10,
                label="Local window radius (px)",
                info="Half-width of the box window used to estimate hand/object depth at each pixel. "
                     "Smaller = more spatial detail but noisier; larger = smoother but less localised. "
                     "Default = dilation radius.",
            )
            object_prompt = gr.Textbox(
                label="Object prompt (for 'Identify Object')",
                placeholder="e.g. ceramic, wood, metal surface",
                interactive=True,
            )
        with gr.Column(scale=2):
            editor = gr.ImageEditor(
                label="Prompt — Object (layer 1)  ·  Hand (layer 2)",
                type="numpy",
                value=_init_frame,
                height=OUT_H,
            )
    with gr.Row():
        prev_btn      = gr.Button("← Prev Folder",            variant="secondary", scale=1)
        reload_btn    = gr.Button("Reload Random Frame",       variant="secondary", scale=1)
        object_btn    = gr.Button("Identify Object",           variant="secondary", scale=1)
        process_btn   = gr.Button("Process Prompt Frame",      variant="primary",   scale=2)
        propagate_btn = gr.Button("Propagate & Save All Frames", variant="secondary", scale=2)
        next_btn      = gr.Button("Next Folder →",             variant="secondary", scale=1)
    with gr.Row():
        viz        = gr.Image(label="Prompt result (blue=object · orange=hand · magenta=touch)", interactive=False, height=OUT_H)
        depth_viz  = gr.Image(label="Depth map",                                                 interactive=False, height=OUT_H)
        object_viz = gr.Image(label="Object detection (blue mask · blue box)",                   interactive=False, height=OUT_H)
    msg = gr.Textbox(label="Messages", interactive=False, lines=5)

    _nav_outputs = [folder_display, editor, status_box, msg, viz, depth_viz]

    reload_btn.click(fn=reload, outputs=[editor, status_box, viz, depth_viz, object_viz])
    object_btn.click(fn=identify_object, inputs=[object_prompt], outputs=[object_viz, msg])
    process_btn.click(
        fn=process_prompt,
        inputs=[editor, dilation, depth_threshold, local_radius_slider],
        outputs=[viz, depth_viz, msg],
    )
    propagate_btn.click(
        fn=propagate_and_save, inputs=[dilation, depth_threshold, local_radius_slider], outputs=[msg]
    )
    next_btn.click(fn=go_next, outputs=_nav_outputs)
    prev_btn.click(fn=go_prev, outputs=_nav_outputs)
