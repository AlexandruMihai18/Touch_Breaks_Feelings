"""Greatest Hits annotation pipeline — frame-level and folder-level entry points.

Pipeline per frame
------------------
1. detect_stick()    — Grounding-DINO + SAM2 → stick mask
2. detect_object()   — SAM2 tip-anchored → object mask
3. compute_depth()   — Depth-Anything-V2 + optional guided filter → depth map
4. compute_touch_region_v2() — pixel-level depth-gap filtering → touch mask
5. save_frame_output()       — write masks + JPEG to disk

annotate_frame() runs steps 1–5 for a single frame.
annotate_folder() loops over frames, writes manifest.json + dataset.json,
and returns a metrics summary string.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from touch_detection_alg.pipeline import compute_depth
from touch_detection_alg.touch import compute_touch_region_v2
from .detection import detect_stick, detect_object
from .io import load_touch_materials, save_frame_output, write_manifest, write_dataset
from .metrics import compute_gh_metrics

_MIN_OBJ_PIXELS = 300


def annotate_frame(
    frame: np.ndarray,
    frame_idx: int,
    stem: str,
    touch_materials: dict[int, str],
    out_dir: Path,
    *,
    dilation: int = 10,
    abs_d_threshold: float = 0.05,
    local_radius: int = 10,
    clahe_clip: float = 0.0,
    guided_filter: bool = True,
    refine_radius: int = 4,
    refine_eps: float = 0.1,
) -> tuple[dict, dict] | tuple[None, str]:
    """Annotate a single Greatest Hits frame.

    Returns (annotation_entry, dataset_row) on success, or (None, skip_reason)
    when the frame is skipped. skip_reason is one of "stick" or "object".

    The depth map is saved alongside the masks even though it is not part of
    the training annotations — it helps debugging the pipeline visually.
    """
    stick_mask, stick_score, _ = detect_stick(frame)
    if stick_score == 0.0:
        return None, "stick"

    obj_mask, tip = detect_object(frame, stick_mask)
    if not tip or np.sum(obj_mask > 0) < _MIN_OBJ_PIXELS:
        return None, "object"

    depth_map = compute_depth(
        frame,
        clahe_clip=clahe_clip,
        guided_filter=guided_filter,
        refine_radius=refine_radius,
        refine_eps=refine_eps,
    )
    touch = compute_touch_region_v2(
        stick_mask, obj_mask, depth_map,
        dilation_radius=dilation,
        abs_d_threshold=abs_d_threshold,
        local_radius=local_radius,
    )

    # Restrict touch to pixels that land on one of the two physical objects.
    valid_region = (stick_mask > 0) | (obj_mask > 0)
    touch = (touch.astype(bool) & valid_region).astype(np.uint8) * 255

    material = touch_materials.get(frame_idx, "")
    paths = save_frame_output(frame, stick_mask, obj_mask, touch, depth_map, out_dir, stem)

    annotation_entry = {
        "image_path":      str(paths["image"]),
        "touch_mask_path": str(paths["touch"]),
        "object_mask_path": str(paths["object"]),
        "stick_mask_path": str(paths["stick"]),
        "depth_map_path":  str(paths["depth"]),
        "frame_index": frame_idx,
        "material": material,
    }
    dataset_row = {
        "image_path":      str(paths["image"]),
        "stick_mask_path": str(paths["stick"]),
        "object_mask_path": str(paths["object"]),
        "target_path":     str(paths["touch"]),
        "depth_path":      str(paths["depth"]),
        "type": "touch" if np.any(touch > 0) else "no-touch",
        "video_id": out_dir.name,
        "material": material,
    }
    return annotation_entry, dataset_row


def annotate_folder(
    folder_name: str,
    frames: list[np.ndarray],
    frame_paths: list[Path],
    masks_root: Path,
    annotations_dir: Path,
    *,
    dilation: int = 10,
    abs_d_threshold: float = 0.05,
    local_radius: int = 10,
    clahe_clip: float = 0.0,
    guided_filter: bool = True,
    refine_radius: int = 4,
    refine_eps: float = 0.1,
) -> str:
    """Batch-annotate a Greatest Hits video folder.

    Iterates over all frames, calls annotate_frame() for each, then writes
    manifest.json + dataset.json and returns a summary string with per-folder
    counts and GT metrics (when annotation JSONs are available).
    """
    touch_materials = load_touch_materials(annotations_dir, folder_name)

    out_dir = masks_root / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_entries: list[dict] = []
    dataset_rows: list[dict] = []
    n_stick_miss = n_obj_miss = 0

    params = dict(
        dilation=dilation,
        abs_d_threshold=abs_d_threshold,
        local_radius=local_radius,
        clahe_clip=clahe_clip,
        guided_filter=guided_filter,
        refine_radius=refine_radius,
        refine_eps=refine_eps,
    )

    for frame, fp in zip(frames, frame_paths):
        frame_idx = int(fp.stem.split("_")[1])
        ann_entry, dataset_row = annotate_frame(
            frame, frame_idx, fp.stem, touch_materials, out_dir, **params
        )
        if ann_entry is None:
            if dataset_row == "stick":
                n_stick_miss += 1
            else:
                n_obj_miss += 1
            continue
        frame_entries.append(ann_entry)
        dataset_rows.append(dataset_row)

    write_manifest(folder_name, frame_entries, out_dir)
    write_dataset(dataset_rows, out_dir)

    n_touch = sum(1 for r in dataset_rows if r["type"] == "touch")
    lines = [
        f"Auto-annotated {len(dataset_rows)}/{len(frames)} frames for '{folder_name}'",
        f"  Stick not detected : {n_stick_miss} frames skipped",
        f"  Object not found   : {n_obj_miss} frames skipped",
        f"  Touch: {n_touch}  No-touch: {len(dataset_rows) - n_touch}",
    ]
    if dataset_rows:
        lines.append(compute_gh_metrics(folder_name, dataset_rows, annotations_dir))
    return "\n".join(lines)
