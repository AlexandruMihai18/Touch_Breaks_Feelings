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
_gh_metadata: dict[tuple[str, int], str] = {}  # (video_id, frame_idx) → label text
_ek_entries_by_video: dict[str, list] = {}      # video_id → list of normalized frame dicts
_folders: list[Path] = sorted(d for d in MASKS_ROOT.iterdir() if d.is_dir())
_folder_idx = [0]
_page_idx = [0]
_samples_raw: list[dict] = []   # original order from annotation file
_samples: list[dict] = []       # displayed order (sorted when coverage sort is on)
_sort_order = ["none"]   # "none" | "desc" (largest first) | "asc" (smallest first)


# ---- coverage helpers ----

def _load_global_coverage() -> dict[str, float]:
    """Parse object_labels_coverage.txt → {object_name: median_coverage}."""
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


def _compute_touch_coverage(touch_mask_path: str) -> float | None:
    """Return nonzero/total for a touch mask file, None if unavailable."""
    p = Path(touch_mask_path) if touch_mask_path else None
    if not p or not p.exists():
        return None
    arr = np.array(Image.open(p).convert("L"))
    return float(np.count_nonzero(arr)) / arr.size if arr.size > 0 else 0.0


def _attach_coverage(samples: list[dict]) -> list[dict]:
    """Attach _coverage to each sample: computed when mask exists, else global median fallback."""
    result = []
    for s in samples:
        s = dict(s)
        cov = _compute_touch_coverage(s.get("touch_mask_path", ""))
        if cov is None:
            cov = _global_coverage.get(s.get("object_name", ""))
        s["_coverage"] = cov  # None means no info available
        result.append(s)
    return result


def _apply_sort() -> None:
    """Populate _samples from _samples_raw, sorting by coverage when enabled."""
    _samples.clear()
    order = _sort_order[0]
    if order in ("asc", "desc"):
        with_cov = _attach_coverage(_samples_raw)
        _samples.extend(
            sorted(
                with_cov,
                key=lambda s: s["_coverage"] if s["_coverage"] is not None else -1.0,
                reverse=(order == "desc"),
            )
        )
    else:
        _samples.extend(_samples_raw)


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
        label  = sample.get("type", "")
        vid    = sample.get("video_id", "")
        obj    = sample.get("object_name", "")
        stem   = Path(sample.get("image_path", "")).stem
        obj_str = f"  {obj}" if obj else ""
        cov = sample.get("_coverage")
        cov_str = f"  cov={cov:.1%}" if cov is not None else ""
        return f"{vid}  [{label}]{obj_str}  {stem}{cov_str}" if label else None

    if not _gh_metadata:
        return None
    p = Path(sample.get("image_path", ""))
    stem = p.stem  # frame_000041
    try:
        frame_idx = int(stem.split("_")[1])
    except (IndexError, ValueError):
        return None
    video_id = p.parent.name
    return _gh_metadata.get((video_id, frame_idx))


def _load_ek_dataset(split: str) -> None:
    """Load EK annotations for the given split and group by video_id."""
    _ek_entries_by_video.clear()
    anno_file = EK_ANNO_DIR / f"{split}.json"
    if not anno_file.exists():
        return
    entries = json.loads(anno_file.read_text())
    for e in entries:
        # Normalize: rename target_path → touch_mask_path for _render_page compatibility
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


def _dilate_touch(touch: np.ndarray, hand: np.ndarray, obj: np.ndarray, radius: int) -> np.ndarray:
    """Dilate touch mask by radius, constrained to the union of hand and object pixels."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    dilated = cv2.dilate(touch, kernel)
    allowed = (hand > 0) | (obj > 0)
    return np.where(allowed, dilated, 0).astype(np.uint8)


def _render_page(
    samples: list, page: int, show_depth: bool = False,
    show_refined: bool = False, dilation: int = 0,
) -> list[np.ndarray | None]:
    """Render PAGE_SIZE frames starting at page * PAGE_SIZE."""
    start = page * PAGE_SIZE
    picks = samples[start : start + PAGE_SIZE]
    out = []
    for s in picks:
        try:
            if show_depth:
                depth_path = s.get("depth_map_path")
                if depth_path and Path(depth_path).exists():
                    out.append(np.array(Image.open(depth_path).convert("RGB")))
                else:
                    out.append(None)
            else:
                img   = np.array(Image.open(s["image_path"]).convert("RGB"))
                obj   = np.array(Image.open(s["object_mask_path"]).convert("L"))
                hand_path = _resolve_hand_path(s)
                if hand_path is None:
                    out.append(None)
                    continue
                hand  = np.array(Image.open(hand_path).convert("L"))
                touch_path = Path(s["touch_mask_path"])
                if show_refined:
                    refined = touch_path.parent / (touch_path.stem + "_refined" + touch_path.suffix)
                    if refined.exists():
                        touch_path = refined
                touch = np.array(Image.open(touch_path).convert("L"))
                if dilation > 0:
                    touch = _dilate_touch(touch, hand, obj, dilation)
                rendered = draw_dual_mask_viz(img, obj, hand, touch, [], [])
                annotation = _annotation_for(s)
                if annotation:
                    rendered = _overlay_annotation(rendered, annotation)
                out.append(rendered)
        except Exception:
            out.append(None)
    while len(out) < PAGE_SIZE:
        out.append(None)
    return out


def _status(samples: list, page: int) -> str:
    if not samples:
        return "No annotations found"
    total = len(samples)
    start = page * PAGE_SIZE + 1
    end = min(start + PAGE_SIZE - 1, total)
    total_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE
    return f"Showing {start}–{end} of {total} frames (page {page + 1}/{total_pages})"


# ---- Gradio callbacks ----


def load_and_sample(idx: int, show_depth: bool = False, show_refined: bool = False, dilation: int = 0):
    _folder_idx[0] = idx
    _page_idx[0] = 0
    name, samples = _load_folder(idx)
    _samples_raw.clear()
    _samples_raw.extend(samples)
    _apply_sort()
    n = len(_folders)
    folder_label = f"Folder {idx + 1}/{n}: {name}"
    if not _samples:
        return (folder_label, "No annotations found", *([None] * PAGE_SIZE))
    imgs = _render_page(_samples, 0, show_depth, show_refined, dilation)
    return (folder_label, _status(_samples, 0), *imgs)


def go_next_folder(show_depth: bool = False, show_refined: bool = False, dilation: int = 0):
    if not _folders:
        return ("No annotation folders", "No annotations", *([None] * PAGE_SIZE))
    return load_and_sample(min(_folder_idx[0] + 1, len(_folders) - 1), show_depth, show_refined, dilation)


def go_prev_folder(show_depth: bool = False, show_refined: bool = False, dilation: int = 0):
    if not _folders:
        return ("No annotation folders", "No annotations", *([None] * PAGE_SIZE))
    return load_and_sample(max(_folder_idx[0] - 1, 0), show_depth, show_refined, dilation)


def go_next_page(show_depth: bool = False, show_refined: bool = False, dilation: int = 0):
    if not _samples:
        return ("No samples loaded", *([None] * PAGE_SIZE))
    total_pages = (len(_samples) + PAGE_SIZE - 1) // PAGE_SIZE
    _page_idx[0] = min(_page_idx[0] + 1, total_pages - 1)
    imgs = _render_page(_samples, _page_idx[0], show_depth, show_refined, dilation)
    return (_status(_samples, _page_idx[0]), *imgs)


def go_prev_page(show_depth: bool = False, show_refined: bool = False, dilation: int = 0):
    if not _samples:
        return ("No samples loaded", *([None] * PAGE_SIZE))
    _page_idx[0] = max(_page_idx[0] - 1, 0)
    imgs = _render_page(_samples, _page_idx[0], show_depth, show_refined, dilation)
    return (_status(_samples, _page_idx[0]), *imgs)


def on_view_change(show_depth: bool, show_refined: bool, dilation: int):
    if not _samples:
        return ("No samples loaded", *([None] * PAGE_SIZE))
    imgs = _render_page(_samples, _page_idx[0], show_depth, show_refined, dilation)
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
    _sort_order[0] = "none"
    split_update    = gr.update(visible=is_ek)
    sort_update     = gr.update(visible=is_ek, value="none")
    refined_update  = gr.update(visible=is_ek, value=False)
    dilation_update = gr.update(visible=is_ek, value=0)
    if not _folders:
        return ("No folders", "No annotations found", split_update, sort_update, refined_update, dilation_update, *([None] * PAGE_SIZE))
    folder_info, status, *imgs = load_and_sample(0)
    return (folder_info, status, split_update, sort_update, refined_update, dilation_update, *imgs)


def switch_ek_split(split: str, show_refined: bool = False, dilation: int = 0):
    """Reload EPIC Kitchen annotations for the selected split."""
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
    return load_and_sample(0, show_refined=show_refined, dilation=dilation)


def on_sort_coverage_change(order: str, show_depth: bool = False, show_refined: bool = False, dilation: int = 0):
    _sort_order[0] = order or "none"
    _page_idx[0] = 0
    _apply_sort()
    if not _samples:
        return (_status(_samples_raw, 0), *([None] * PAGE_SIZE))
    imgs = _render_page(_samples, 0, show_depth, show_refined, dilation)
    return (_status(_samples, 0), *imgs)


# ---- initialize ----

if _folders:
    _init_info, _init_status, *_init_imgs = load_and_sample(0)
else:
    _init_info    = "No annotation folders"
    _init_status  = "No annotations found"
    _init_imgs    = [None] * PAGE_SIZE


# ---- UI ----

with gr.Blocks() as tab:
    gr.Markdown(
        "## Review Annotations\n"
        "Browse 10 frames per page. "
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
        status_box = gr.Textbox(label="Status", value=_init_status, interactive=False)
    with gr.Row():
        prev_folder_btn = gr.Button("← Prev Folder", variant="secondary")
        next_folder_btn = gr.Button("Next Folder →", variant="secondary")
        show_depth_cb = gr.Checkbox(label="Show depth maps", value=False)
        sort_coverage_radio = gr.Radio(
            choices=[("None", "none"), ("Largest first ↓", "desc"), ("Smallest first ↑", "asc")],
            value="none",
            label="Sort by coverage",
            visible=False,
            info="'Smallest first' puts tiny touch regions at the top — useful for checking dilation.",
        )
        show_refined_cb = gr.Checkbox(
            label="Show refined masks",
            value=False,
            visible=False,
            info="Load *_touch_refined.png (depth-filtered). Falls back to original when unavailable.",
        )
    with gr.Row():
        dilation_slider = gr.Slider(
            minimum=0, maximum=40, step=1, value=0,
            label="Live dilation (px)",
            visible=False,
            info="Extra dilation applied on-the-fly, constrained to hand ∪ object pixels. "
                 "Use this to preview before running scripts/dilate_touch_masks.py at scale.",
        )
    with gr.Row():
        prev_page_btn = gr.Button("← Prev 10", variant="secondary")
        next_page_btn = gr.Button("Next 10 →", variant="secondary")

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

    _page_outputs   = [status_box] + img_components
    _folder_outputs = [folder_info, status_box] + img_components
    _switch_outputs = [folder_info, status_box, ek_split_radio, sort_coverage_radio, show_refined_cb, dilation_slider] + img_components
    _view_inputs    = [show_depth_cb, show_refined_cb, dilation_slider]

    dataset_radio.change(
        fn=switch_dataset,
        inputs=[dataset_radio, ek_split_radio],
        outputs=_switch_outputs,
    )
    ek_split_radio.change(
        fn=switch_ek_split,
        inputs=[ek_split_radio, show_refined_cb, dilation_slider],
        outputs=_folder_outputs,
    )
    prev_folder_btn.click(fn=go_prev_folder, inputs=_view_inputs, outputs=_folder_outputs)
    next_folder_btn.click(fn=go_next_folder, inputs=_view_inputs, outputs=_folder_outputs)
    prev_page_btn.click(fn=go_prev_page, inputs=_view_inputs, outputs=_page_outputs)
    next_page_btn.click(fn=go_next_page, inputs=_view_inputs, outputs=_page_outputs)
    show_depth_cb.change(fn=on_view_change, inputs=_view_inputs, outputs=_page_outputs)
    show_refined_cb.change(fn=on_view_change, inputs=_view_inputs, outputs=_page_outputs)
    dilation_slider.change(fn=on_view_change, inputs=_view_inputs, outputs=_page_outputs)
    sort_coverage_radio.change(
        fn=on_sort_coverage_change,
        inputs=[sort_coverage_radio] + _view_inputs,
        outputs=_page_outputs,
    )
