"""File I/O for the Greatest Hits annotation pipeline."""

import json
from pathlib import Path

import numpy as np
from PIL import Image



def load_touch_materials(annotations_dir: Path, video_id: str) -> dict[int, str]:
    """Read train.json + val.json and return {frame_idx: material} for touch frames in video_id.

    Returns an empty dict when no annotation files exist (non-fatal — annotation
    proceeds without material labels).
    """
    result: dict[int, str] = {}
    for split in ("train", "val"):
        path = annotations_dir / f"{split}.json"
        if not path.exists():
            continue
        for entry in json.loads(path.read_text()):
            if entry.get("video_id") != video_id:
                continue
            material = entry.get("material")
            frame_idx = entry.get("frame_idx")
            if material and frame_idx is not None:
                result[int(frame_idx)] = str(material)
    return result


def save_frame_output(
    frame: np.ndarray,
    stick_mask: np.ndarray,
    obj_mask: np.ndarray,
    touch: np.ndarray,
    depth_map: np.ndarray,
    out_dir: Path,
    stem: str,
) -> dict[str, Path]:
    """Save all mask outputs for one annotated frame. Returns a dict of written paths."""
    paths = {
        "touch":  out_dir / f"{stem}_touch.png",
        "object": out_dir / f"{stem}_object.png",
        "stick":  out_dir / f"{stem}_stick.png",
        "depth":  out_dir / f"{stem}_depth.png",
        "image":  out_dir / f"{stem}.jpg",
    }
    Image.fromarray((touch > 0).astype(np.uint8) * 255, mode="L").save(paths["touch"])
    Image.fromarray((obj_mask > 0).astype(np.uint8) * 255, mode="L").save(paths["object"])
    Image.fromarray((stick_mask > 0).astype(np.uint8) * 255, mode="L").save(paths["stick"])
    Image.fromarray((depth_map * 65535).astype(np.uint16)).save(paths["depth"])
    Image.fromarray(frame).save(paths["image"], quality=95)
    return paths


def write_manifest(folder_name: str, frame_entries: list[dict], out_dir: Path) -> None:
    """Write manifest.json listing all annotated frames for a folder."""
    with open(out_dir / "manifest.json", "w") as f:
        json.dump({"video_id": folder_name, "frames": frame_entries}, f, indent=2)


def write_dataset(dataset_rows: list[dict], out_dir: Path) -> None:
    """Write dataset.json with one record per annotated frame."""
    with open(out_dir / "dataset.json", "w") as f:
        json.dump(dataset_rows, f, indent=2)
