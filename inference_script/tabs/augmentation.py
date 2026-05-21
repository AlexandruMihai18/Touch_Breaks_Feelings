import json
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from PIL import ImageDraw, ImageFont

from inference_script.shared import (DATASET_NAMES, DATASETS, EK_ANNO_DIR,
                                      EK_COVERAGE_PATH, GH_ANNO_DIR,
                                      MASKS_ROOT, OUT_H)
from inference_script.src.visualization import draw_dual_mask_viz

PAGE_SIZE = 10

# ---- state ----

_dataset_name = [DATASET_NAMES[0]]
_masks_root = [MASKS_ROOT]
_gh_metadata: dict[tuple[str, int], str] = {}
_ek_entries_by_video: dict[str, list] = {}
_folders: list[Path] = sorted(d for d in MASKS_ROOT.iterdir() if d.is_dir())
_folder_idx = [0]
_page_idx = [0]
_samples_raw: list[dict] = []
_samples: list[dict] = []


# ---- helpers ----

def _load_gh_metadata() -> dict[tuple[str, int], str]:
    lookup: dict[tuple[str, int], str] = {}
    for split in ("train", "val"):
        path = GH_ANNO_DIR / f"{split}.json"
        if not path.exists():
            continue
        for entry in json.loads(path.read_text()):
            frame_idx = entry.get("frame_idx")
            if frame_idx is None:
                continue
            parts = [p for p in (entry.get("material"), entry.get("action")) if p]
            lookup[(entry["video_id"], int(frame_idx))] = " · ".join(parts)
    return lookup


_FONT: ImageFont.ImageFont | None = None


def _get_font() -> ImageFont.ImageFont:
    global _FONT
    if _FONT is None:
        for path in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ):
            try:
                _FONT = ImageFont.truetype(path, 70)
                break
            except Exception:
                pass
        if _FONT is None:
            _FONT = ImageFont.load_default()
    return _FONT


def _overlay_annotation(img: np.ndarray, text: str) -> np.ndarray:
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    font = _get_font()
    draw.text((4, 4), text, fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    return np.array(pil)


def _annotation_for(sample: dict) -> str | None:
    if _dataset_name[0] == "EPIC Kitchen":
        label = sample.get("type", "")
        vid   = sample.get("video_id", "")
        obj   = sample.get("object_name", "")
        stem  = Path(sample.get("image_path", "")).stem
        obj_str = f"  {obj}" if obj else ""
        return f"{vid}  [{label}]{obj_str}  {stem}" if label else None

    if not _gh_metadata:
        return None
    p = Path(sample.get("image_path", ""))
    stem = p.stem
    try:
        frame_idx = int(stem.split("_")[1])
    except (IndexError, ValueError):
        return None
    video_id = p.parent.name
    return _gh_metadata.get((video_id, frame_idx))


def _load_ek_dataset(split: str) -> None:
    _ek_entries_by_video.clear()
    anno_file = EK_ANNO_DIR / f"{split}.json"
    if not anno_file.exists():
        return
    entries = json.loads(anno_file.read_text())
    for e in entries:
        norm = dict(e)
        if "target_path" in norm and "touch_mask_path" not in norm:
            norm["touch_mask_path"] = norm.pop("target_path")
        _ek_entries_by_video.setdefault(e["video_id"], []).append(norm)


def _load_folder(idx: int) -> tuple[str, list]:
    if not (0 <= idx < len(_folders)):
        return "No folders", []
    folder = _folders[idx]
    if _dataset_name[0] == "EPIC Kitchen":
        vid     = folder.name
        entries = _ek_entries_by_video.get(vid, [])
        return vid, entries
    manifest = folder / "manifest.json"
    if not manifest.exists():
        return folder.name, []
    with open(manifest) as f:
        return folder.name, json.load(f).get("frames", [])


# ---- augmentation helpers ----

_SCALE_OPTIONS = {"100% (original)": 1.0, "75%": 0.75, "50%": 0.50, "25%": 0.25}


def _resolve_hand_path(s: dict) -> str | None:
    """Return the hand/stick mask path that actually exists on disk.

    Some GH manifests list hand_mask_path but only _stick.png files were written,
    and vice-versa. Try both keys, then fall back to deriving from image_path stem.
    """
    for key in ("hand_mask_path", "stick_mask_path"):
        if key in s and Path(s[key]).exists():
            return s[key]
    img_stem = Path(s.get("image_path", "")).stem
    folder = Path(s.get("image_path", "")).parent
    for suffix in ("_stick.png", "_hand.png"):
        candidate = folder / (img_stem + suffix)
        if candidate.exists():
            return str(candidate)
    return None


def _simulate_resolution(arr: np.ndarray, scale: float, out_h: int | None = None) -> np.ndarray:
    """Downscale then upscale to simulate lower resolution.

    If out_h is given, the final upsample targets that height (preserving aspect)
    using INTER_NEAREST so pixel blocks survive without smoothing.
    """
    if scale == 1.0 and out_h is None:
        return arr
    h, w = arr.shape[:2]
    small_w, small_h = max(1, int(w * scale)), max(1, int(h * scale))
    small = cv2.resize(arr, (small_w, small_h), interpolation=cv2.INTER_AREA)
    if out_h is not None:
        out_w = max(1, int(w * out_h / h))
        return cv2.resize(small, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def _apply_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    k = max(3, int(6 * sigma + 1) | 1)  # odd kernel, at least 3
    return cv2.GaussianBlur(img, (k, k), sigma)


def _render_page(
    samples: list,
    page: int,
    scale_label: str = "100% (original)",
    sigma: float = 0,
) -> list[np.ndarray | None]:
    scale = _SCALE_OPTIONS.get(scale_label, 1.0)

    start = page * PAGE_SIZE
    picks = samples[start : start + PAGE_SIZE]
    out = []
    for s in picks:
        try:
            img   = np.array(Image.open(s["image_path"]).convert("RGB"))
            obj   = np.array(Image.open(s["object_mask_path"]).convert("L"))
            hand_path = _resolve_hand_path(s)
            if hand_path is None:
                out.append(None)
                continue
            hand  = np.array(Image.open(hand_path).convert("L"))
            touch = np.array(Image.open(s["touch_mask_path"]).convert("L"))

            orig_h, orig_w = img.shape[:2]

            img = _apply_blur(img, sigma)

            rendered = draw_dual_mask_viz(img, obj, hand, touch, [], [])

            # Pixelate the full rendered image and pre-scale to OUT_H with
            # INTER_NEAREST so Gradio receives a correctly-sized image and
            # cannot smooth away the pixel blocks with its own interpolation.
            rendered = _simulate_resolution(rendered, scale, out_h=OUT_H)

            # Overlay effective resolution so the downscale factor is explicit
            eff_h, eff_w = int(orig_h * scale), int(orig_w * scale)
            res_text = f"{eff_w}×{eff_h}px"
            annotation = _annotation_for(s)
            label = f"{res_text}  {annotation}" if annotation else res_text
            rendered = _overlay_annotation(rendered, label)
            out.append(rendered)
        except Exception:
            out.append(None)
    while len(out) < PAGE_SIZE:
        out.append(None)
    return out


_SAVE_ROOT = Path(__file__).parents[2] / "data" / "augmentation_previews"


def save_frames(scale_label: str = "100% (original)", sigma: float = 0) -> str:
    if not _samples:
        return "No samples loaded — nothing to save."
    scale = _SCALE_OPTIONS.get(scale_label, 1.0)
    scale_tag = scale_label.replace("% (original)", "pct").replace("%", "pct").replace(" ", "")
    blur_tag  = f"blur{sigma:.1f}".replace(".", "p")
    out_dir = _SAVE_ROOT / f"page{_page_idx[0] + 1}_{scale_tag}_{blur_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = _render_page(_samples, _page_idx[0], scale_label, sigma)
    start = _page_idx[0] * PAGE_SIZE
    saved = 0
    for i, (s, arr) in enumerate(zip(_samples[start: start + PAGE_SIZE], imgs)):
        if arr is None:
            continue
        stem = Path(s.get("image_path", f"frame_{i:03d}")).stem
        fname = f"{i + 1:02d}_{stem}_{scale_tag}_{blur_tag}.png"
        Image.fromarray(arr).save(out_dir / fname)
        saved += 1
    return f"Saved {saved} frame(s) → {out_dir}"


def _status(samples: list, page: int) -> str:
    if not samples:
        return "No annotations found"
    total = len(samples)
    start = page * PAGE_SIZE + 1
    end = min(start + PAGE_SIZE - 1, total)
    total_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE
    return f"Showing {start}–{end} of {total} frames (page {page + 1}/{total_pages})"


# ---- Gradio callbacks ----

def load_and_sample(idx: int, scale_label: str = "100% (original)", sigma: float = 0):
    _folder_idx[0] = idx
    _page_idx[0] = 0
    name, samples = _load_folder(idx)
    _samples_raw.clear()
    _samples_raw.extend(samples)
    _samples.clear()
    _samples.extend(_samples_raw)
    n = len(_folders)
    folder_label = f"Folder {idx + 1}/{n}: {name}"
    if not _samples:
        return (folder_label, "No annotations found", *([None] * PAGE_SIZE))
    imgs = _render_page(_samples, 0, scale_label, sigma)
    return (folder_label, _status(_samples, 0), *imgs)


def go_next_folder(scale_label: str = "100% (original)", sigma: float = 0):
    if not _folders:
        return ("No annotation folders", "No annotations", *([None] * PAGE_SIZE))
    return load_and_sample(min(_folder_idx[0] + 1, len(_folders) - 1), scale_label, sigma)


def go_prev_folder(scale_label: str = "100% (original)", sigma: float = 0):
    if not _folders:
        return ("No annotation folders", "No annotations", *([None] * PAGE_SIZE))
    return load_and_sample(max(_folder_idx[0] - 1, 0), scale_label, sigma)


def go_next_page(scale_label: str = "100% (original)", sigma: float = 0):
    if not _samples:
        return ("No samples loaded", *([None] * PAGE_SIZE))
    total_pages = (len(_samples) + PAGE_SIZE - 1) // PAGE_SIZE
    _page_idx[0] = min(_page_idx[0] + 1, total_pages - 1)
    imgs = _render_page(_samples, _page_idx[0], scale_label, sigma)
    return (_status(_samples, _page_idx[0]), *imgs)


def go_prev_page(scale_label: str = "100% (original)", sigma: float = 0):
    if not _samples:
        return ("No samples loaded", *([None] * PAGE_SIZE))
    _page_idx[0] = max(_page_idx[0] - 1, 0)
    imgs = _render_page(_samples, _page_idx[0], scale_label, sigma)
    return (_status(_samples, _page_idx[0]), *imgs)


def on_aug_change(scale_label: str, sigma: float):
    if not _samples:
        return ("No samples loaded", *([None] * PAGE_SIZE))
    imgs = _render_page(_samples, _page_idx[0], scale_label, sigma)
    return (_status(_samples, _page_idx[0]), *imgs)


def switch_dataset(dataset_name: str, ek_split: str):
    _dataset_name[0] = dataset_name
    is_ek = dataset_name == "EPIC Kitchen"

    _, masks_root = DATASETS[dataset_name]
    _masks_root[0] = masks_root
    _folders.clear()
    _ek_entries_by_video.clear()
    _gh_metadata.clear()

    if is_ek:
        _load_ek_dataset(ek_split)
        ek_masks = masks_root
        for vid in sorted(_ek_entries_by_video.keys()):
            _folders.append(ek_masks / vid)
    else:
        if masks_root.exists():
            _folders.extend(sorted(d for d in masks_root.iterdir() if d.is_dir()))
        if dataset_name == "Greatest Hits":
            _gh_metadata.update(_load_gh_metadata())

    _folder_idx[0] = 0
    _page_idx[0] = 0
    _samples_raw.clear()
    _samples.clear()
    split_update = gr.update(visible=is_ek)
    if not _folders:
        return ("No folders", "No annotations found", split_update, *([None] * PAGE_SIZE))
    folder_info, status, *imgs = load_and_sample(0)
    return (folder_info, status, split_update, *imgs)


def switch_ek_split(split: str, scale_label: str = "100% (original)", sigma: float = 0):
    _load_ek_dataset(split)
    _folders.clear()
    ek_masks = _masks_root[0]
    for vid in sorted(_ek_entries_by_video.keys()):
        _folders.append(ek_masks / vid)
    _folder_idx[0] = 0
    _page_idx[0] = 0
    _samples_raw.clear()
    _samples.clear()
    if not _folders:
        return ("No folders", "No annotations found", *([None] * PAGE_SIZE))
    return load_and_sample(0, scale_label, sigma)


# ---- initialize ----

if _folders:
    _init_info, _init_status, *_init_imgs = load_and_sample(0)
else:
    _init_info   = "No annotation folders"
    _init_status = "No annotations found"
    _init_imgs   = [None] * PAGE_SIZE


# ---- UI ----

with gr.Blocks() as tab:
    gr.Markdown(
        "## Augmentation Preview\n"
        "Visualize how size reduction and Gaussian blurring affect frames and masks.\n"
        "**Blue** = object · **Orange** = hand · **Magenta** = touch"
    )
    with gr.Row():
        dataset_radio = gr.Radio(
            choices=DATASET_NAMES,
            value=DATASET_NAMES[0],
            label="Dataset",
            interactive=True,
        )
        ek_split_radio = gr.Radio(
            choices=["train", "val"],
            value="train",
            label="Split",
            visible=False,
            interactive=True,
        )
    with gr.Row():
        folder_info = gr.Textbox(label="Folder", value=_init_info, interactive=False)
        status_box  = gr.Textbox(label="Status",  value=_init_status, interactive=False)
    with gr.Row():
        prev_folder_btn = gr.Button("← Prev Folder", variant="secondary")
        next_folder_btn = gr.Button("Next Folder →",  variant="secondary")
    with gr.Row():
        scale_radio = gr.Radio(
            choices=list(_SCALE_OPTIONS.keys()),
            value="100% (original)",
            label="Image / mask size",
            info="Downscales both the RGB frame and all masks before rendering.",
            interactive=True,
        )
        blur_slider = gr.Slider(
            minimum=0, maximum=10, step=0.5, value=0,
            label="Gaussian blur σ",
            info="Applied to the RGB image only. 0 = no blur. Can be combined with size reduction.",
        )
    with gr.Row():
        prev_page_btn = gr.Button("← Prev 10", variant="secondary")
        next_page_btn = gr.Button("Next 10 →",  variant="secondary")
        save_btn      = gr.Button("Save frames", variant="primary")
    with gr.Row():
        save_status = gr.Textbox(label="Save status", interactive=False, value="")

    img_components = []
    for row in range(2):
        with gr.Row():
            for col in range(5):
                img_components.append(
                    gr.Image(
                        label="",
                        value=_init_imgs[row * 5 + col],
                        interactive=False,
                        height=OUT_H,
                    )
                )

    _aug_inputs     = [scale_radio, blur_slider]
    _page_outputs   = [status_box] + img_components
    _folder_outputs = [folder_info, status_box] + img_components
    _switch_outputs = [folder_info, status_box, ek_split_radio] + img_components

    dataset_radio.change(
        fn=switch_dataset,
        inputs=[dataset_radio, ek_split_radio],
        outputs=_switch_outputs,
    )
    ek_split_radio.change(
        fn=switch_ek_split,
        inputs=[ek_split_radio] + _aug_inputs,
        outputs=_folder_outputs,
    )
    prev_folder_btn.click(fn=go_prev_folder, inputs=_aug_inputs, outputs=_folder_outputs)
    next_folder_btn.click(fn=go_next_folder, inputs=_aug_inputs, outputs=_folder_outputs)
    prev_page_btn.click(fn=go_prev_page, inputs=_aug_inputs, outputs=_page_outputs)
    next_page_btn.click(fn=go_next_page, inputs=_aug_inputs, outputs=_page_outputs)
    scale_radio.change(fn=on_aug_change,  inputs=_aug_inputs, outputs=_page_outputs)
    blur_slider.change(fn=on_aug_change, inputs=_aug_inputs, outputs=_page_outputs)
    save_btn.click(fn=save_frames, inputs=_aug_inputs, outputs=[save_status])
