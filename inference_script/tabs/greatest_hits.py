"""Greatest Hits annotation tab — stick-anchored workflow."""

import random
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from inference_script.shared import GH_ANNO_DIR, GH_FRAMES_ROOT, GH_MASKS_ROOT, OUT_H
from touch_detection_alg.depth import (calibrate_blur_thresholds,
                                         colorize_depth, predict_depth,
                                         sharpen_depth)
from touch_detection_alg.greatest_hits.detection import (detect_hand_center,
                                                          find_stick_endpoints,
                                                          run_sam_from_stick_tip,
                                                          run_sam_text)
from touch_detection_alg.touch import compute_touch_region_v2
from inference_script.src.visualization import (draw_dual_mask_viz,
                                                draw_endpoints_debug_viz,
                                                draw_stick_viz,
                                                draw_tip_detection_viz)
from touch_detection_alg.greatest_hits import annotate_folder

# ---- state ----

_folders: list[Path] = (
    sorted(d for d in GH_FRAMES_ROOT.iterdir() if d.is_dir())
    if GH_FRAMES_ROOT.exists()
    else []
)
_folder_idx = [0]
_frames: list[np.ndarray] = []
_frame_paths: list[Path] = []
_masks: dict = {}
_prompt_idx = [0]

_STICK_PROMPT = "thin wooden stick."


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
    return frame, status, None, None


def identify_stick():
    if not _frames:
        return None, "No frame loaded"
    frame = _frames[_prompt_idx[0]]
    mask, box, score, sharpened = run_sam_text(frame, _STICK_PROMPT)
    if score == 0.0:
        return draw_stick_viz(sharpened, mask, box), "No drum stick detected — try reloading a different frame"
    _masks["stick_mask"] = mask
    _masks["stick_box"] = box
    stem = _frame_paths[_prompt_idx[0]].stem if _frame_paths else "?"
    return draw_stick_viz(sharpened, mask, box), f"Stick detected in '{stem}' — score: {score:.3f} | box: {[round(v, 1) for v in box]}"


def identify_object_from_tip():
    if not _frames:
        return None, "No frame loaded"
    if "stick_mask" not in _masks:
        return None, "Run 'Identify Stick' first to get the stick mask"
    frame = _frames[_prompt_idx[0]]
    stick_mask = _masks["stick_mask"]
    obj_mask, tip = run_sam_from_stick_tip(frame, stick_mask)
    stem = _frame_paths[_prompt_idx[0]].stem if _frame_paths else "?"
    if not tip or not np.any(obj_mask > 0):
        return (
            draw_tip_detection_viz(frame, stick_mask, obj_mask, tip or None),
            f"No object found at stick tip in '{stem}'",
        )
    _masks["obj_from_tip_mask"] = obj_mask
    px = int(np.sum(obj_mask > 0))
    return draw_tip_detection_viz(frame, stick_mask, obj_mask, tip), f"Object found at tip {tip} in '{stem}' — {px} px"


def debug_endpoints():
    if not _frames:
        return None, "No frame loaded"
    if "stick_mask" not in _masks:
        return None, "Run 'Identify Stick' first to get the stick mask"
    frame = _frames[_prompt_idx[0]]
    stick_mask = _masks["stick_mask"]
    hand_center, hand_score = detect_hand_center(frame, stick_mask)
    ep_a, ep_b, _, tip_is_ep_a = find_stick_endpoints(stick_mask, hand_center, hand_score)
    hand_str = (
        f"hand @ {[round(v, 1) for v in hand_center]} (conf={hand_score:.2f})"
        if hand_center else "no hand detected"
    )
    stem = _frame_paths[_prompt_idx[0]].stem if _frame_paths else "?"
    if not ep_a:
        return draw_stick_viz(frame, stick_mask), f"Too few stick pixels in '{stem}' | {hand_str}"
    tip_label = "A" if tip_is_ep_a else "B"
    tip = ep_a if tip_is_ep_a else ep_b
    viz = draw_endpoints_debug_viz(frame, stick_mask, ep_a, ep_b, tip_is_ep_a, hand_center)
    return viz, f"'{stem}' | ep_a={ep_a}  ep_b={ep_b}  tip={tip_label} @ {tip} | {hand_str}"


def process_frame_from_tip(dilation, abs_d_threshold, local_radius, guided_filter, refine_radius, refine_eps, clahe_on, clahe_clip):
    if not _frames:
        return None, None, "No frame loaded"
    if "stick_mask" not in _masks:
        return None, None, "Run 'Identify Stick' first"
    if "obj_from_tip_mask" not in _masks:
        return None, None, "Run 'Object from Tip' first"
    frame = _frames[_prompt_idx[0]]
    stick_mask = _masks["stick_mask"]
    obj_mask = _masks["obj_from_tip_mask"]
    depth_map = predict_depth(frame, clahe_clip=float(clahe_clip) if clahe_on else 0.0)
    if guided_filter:
        depth_map = sharpen_depth(depth_map, frame, radius=int(refine_radius), eps=float(refine_eps))
    touch = compute_touch_region_v2(
        obj_mask, stick_mask, depth_map,
        dilation_radius=int(dilation),
        abs_d_threshold=float(abs_d_threshold),
        local_radius=int(local_radius),
    )
    valid_region = (obj_mask > 0) | (stick_mask > 0)
    touch = (touch.astype(bool) & valid_region).astype(np.uint8) * 255
    stem = _frame_paths[_prompt_idx[0]].stem if _frame_paths else "?"
    n_touch_px = int(np.sum(touch > 0))
    status = "touch detected" if n_touch_px > 0 else "no touch"
    return (
        draw_dual_mask_viz(frame, obj_mask, stick_mask, touch, [], []),
        colorize_depth(depth_map),
        f"Frame '{stem}' processed — {status} ({n_touch_px} px)",
    )


def auto_annotate(dilation, abs_d_threshold, local_radius, guided_filter, refine_radius, refine_eps, clahe_on, clahe_clip):
    if not _folders:
        return "No folders loaded"
    return annotate_folder(
        folder_name=_folders[_folder_idx[0]].name,
        frames=_frames,
        frame_paths=_frame_paths,
        masks_root=GH_MASKS_ROOT,
        annotations_dir=GH_ANNO_DIR,
        dilation=int(dilation),
        abs_d_threshold=float(abs_d_threshold),
        local_radius=int(local_radius),
        guided_filter=bool(guided_filter),
        refine_radius=int(refine_radius),
        refine_eps=float(refine_eps),
        clahe_clip=float(clahe_clip) if clahe_on else 0.0,
    )


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
    _init_status = "No frame folders found in data/greatest_hits/frames"


# ---- UI ----

with gr.Blocks() as tab:
    gr.Markdown(
        "## Greatest Hits Annotation\n"
        "Stick-anchored auto-annotation for the Greatest Hits dataset. "
        "Use **Identify Stick** → **Object from Tip** to verify detection on a frame, "
        "then **Auto-Annotate** to batch-process the current folder."
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
                minimum=5, maximum=100, value=10, step=5,
                label="Touch region dilation (px)",
            )
            abs_d_threshold = gr.Slider(
                minimum=0.01, maximum=0.30, value=0.05, step=0.01,
                label="Abs. depth gap threshold",
            )
            local_radius = gr.Slider(
                minimum=2, maximum=30, value=10, step=2,
                label="Local depth window (px)",
            )
            clahe_on = gr.Checkbox(
                value=True, label="CLAHE contrast enhancement",
            )
            clahe_clip = gr.Slider(
                minimum=0.5, maximum=8.0, value=2.0, step=0.5,
                label="CLAHE clip limit",
            )
            guided_filter = gr.Checkbox(
                value=True, label="Guided depth filter",
            )
            with gr.Row():
                refine_radius = gr.Slider(
                    minimum=1, maximum=10, value=4, step=1,
                    label="Filter radius (px)",
                )
                refine_eps = gr.Slider(
                    minimum=0.01, maximum=1.0, value=0.1, step=0.01,
                    label="Filter ε",
                )
        with gr.Column(scale=2):
            frame_img = gr.Image(
                label="Current frame",
                type="numpy",
                value=_init_frame,
                height=OUT_H,
                interactive=False,
            )
    with gr.Row():
        prev_btn      = gr.Button("← Prev Folder",        variant="secondary", scale=1)
        reload_btn    = gr.Button("Reload Random Frame",   variant="secondary", scale=1)
        stick_btn     = gr.Button("Identify Stick",        variant="secondary", scale=1)
        tip_btn       = gr.Button("Object from Tip",       variant="secondary", scale=1)
        debug_ep_btn  = gr.Button("Debug Endpoints",       variant="secondary", scale=1)
        frame_tip_btn = gr.Button("Process Frame",         variant="primary",   scale=1)
        auto_btn      = gr.Button("Auto-Annotate",         variant="primary",   scale=2)
        next_btn      = gr.Button("Next Folder →",         variant="secondary", scale=1)
    with gr.Row():
        stick_viz  = gr.Image(label="Stick detection (yellow mask · yellow box)",              interactive=False, height=OUT_H)
        object_viz = gr.Image(label="Object at tip",                                           interactive=False, height=OUT_H)
        touch_viz  = gr.Image(label="Touch result (blue=object · orange=stick · magenta=touch)", interactive=False, height=OUT_H)
        depth_viz  = gr.Image(label="Depth map",                                               interactive=False, height=OUT_H)
    msg = gr.Textbox(label="Messages", interactive=False, lines=5)

    _nav_outputs = [folder_display, frame_img, status_box, msg, stick_viz, object_viz]

    reload_btn.click(fn=reload, outputs=[frame_img, status_box, stick_viz, object_viz])
    stick_btn.click(fn=identify_stick, outputs=[stick_viz, msg])
    tip_btn.click(fn=identify_object_from_tip, outputs=[object_viz, msg])
    debug_ep_btn.click(fn=debug_endpoints, outputs=[object_viz, msg])
    _params = [dilation, abs_d_threshold, local_radius, guided_filter, refine_radius, refine_eps, clahe_on, clahe_clip]
    frame_tip_btn.click(
        fn=process_frame_from_tip,
        inputs=_params,
        outputs=[touch_viz, depth_viz, msg],
    )
    auto_btn.click(fn=auto_annotate, inputs=_params, outputs=[msg])
    next_btn.click(fn=go_next, outputs=_nav_outputs)
    prev_btn.click(fn=go_prev, outputs=_nav_outputs)
