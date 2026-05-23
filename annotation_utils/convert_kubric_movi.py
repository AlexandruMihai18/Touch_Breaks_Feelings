"""
Convert a local Kubric MOVi TFDS split into the touch_from_segmentation layout.

MOVi-A contains rigid-body object collisions, not hands.  This converter maps
each unambiguous frame with one visible object-object collision pair into one
training pair:

    colliding instance A -> object1_mask_path (role: colliding_object_1)
    colliding instance B -> object2_mask_path (role: colliding_object_2)
    visible collision image point -> target_path touch disk clipped to visible object pixels

It also writes frame-level "no-touch" rows sampled from frames with no valid
object-object collision anywhere in the scene.

Output layout:

    data/kubric_movi_a/
        frames/{video_id}/frame_000012.jpg
        masks/{video_id}/frame_000012_depth.png
        masks/{video_id}/frame_000012_object1.png
        masks/{video_id}/frame_000012_object2.png
        masks/{video_id}/frame_000012_touch.png
        annotations/val.json
        annotations/val_ctx_index.json

Example:

    python annotation_utils/convert_kubric_movi.py \\
        --data_dir data/kubric_tfds \\
        --dataset movi_a/128x128 \\
        --split validation \\
        --output_root data/kubric_movi_a \\
        --max_videos 10
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


FLOOR_OR_BACKGROUND_INSTANCE = 65535
LABEL_NAMES = {
    "color_label": ["blue", "brown", "cyan", "gray", "green", "purple", "red", "yellow"],
    "material_label": ["metal", "rubber"],
    "shape_label": ["cube", "cylinder", "sphere"],
    "size_label": ["small", "large"],
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _require_tfds() -> Any:
    try:
        import tensorflow_datasets as tfds
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "tensorflow-datasets is required to read Kubric TFDS files.\n"
            "Install it in a Python 3.11 environment, for example:\n"
            "  conda create -n kubric_movi python=3.11 -y\n"
            "  conda activate kubric_movi\n"
            "  python -m pip install tensorflow tensorflow-datasets tqdm pillow"
        ) from exc
    return tfds


def _decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode_text(value.item())
    return str(value)


def _as_int(value: Any) -> int:
    if isinstance(value, np.ndarray):
        return int(value.item())
    return int(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _binary_mask(mask: np.ndarray) -> Image.Image:
    return Image.fromarray((mask.astype(np.uint8) * 255), mode="L")


def _depth_image(depth: np.ndarray) -> Image.Image:
    depth = np.asarray(depth)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2D depth frame, got shape {depth.shape}")

    if np.issubdtype(depth.dtype, np.integer):
        depth_uint16 = np.clip(depth, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    else:
        depth_float = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        depth_float = np.maximum(depth_float, 0.0)
        if depth_float.size and float(depth_float.max()) <= 1.0:
            depth_uint16 = np.round(depth_float * np.iinfo(np.uint16).max).astype(np.uint16)
        else:
            depth_uint16 = np.clip(
                np.round(depth_float * 1000.0),
                0,
                np.iinfo(np.uint16).max,
            ).astype(np.uint16)
    return Image.fromarray(depth_uint16, mode="I;16")


def _touch_disk(
    image_position: np.ndarray,
    height: int,
    width: int,
    radius: int,
) -> Image.Image:
    """Create a binary touch mask around normalized (x, y) image coordinates."""
    x = float(image_position[0]) * (width - 1)
    y = float(image_position[1]) * (height - 1)
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
    return mask


def _visible_touch_mask(
    image_position: np.ndarray,
    height: int,
    width: int,
    radius: int,
    object1_mask: np.ndarray,
    object2_mask: np.ndarray,
) -> np.ndarray | None:
    """Return the visible part of a touch disk, or None if the contact is occluded."""
    touch_mask = np.asarray(_touch_disk(image_position, height, width, radius)) > 0
    touch_on_object1 = touch_mask & object1_mask
    touch_on_object2 = touch_mask & object2_mask
    if not touch_on_object1.any() or not touch_on_object2.any():
        return None
    return touch_mask & (object1_mask | object2_mask)


def _mask_has_content(path: str) -> bool:
    try:
        return Image.open(path).convert("L").getbbox() is not None
    except Exception:
        return False


def _build_ctx_index(entries: list[dict[str, Any]]) -> dict[str, list[int]]:
    index: dict[str, list[int]] = defaultdict(list)
    for i, entry in enumerate(entries):
        if (
            _mask_has_content(entry.get("object1_mask_path", ""))
            and _mask_has_content(entry.get("object2_mask_path", ""))
        ):
            index[entry.get("object_name") or entry["video_id"]].append(i)
    return dict(index)


def _iter_collision_events(sample: dict[str, Any]) -> list[dict[str, Any]]:
    collisions = sample["events"]["collisions"]
    if isinstance(collisions, dict):
        n = len(collisions["frame"])
        return [{key: collisions[key][i] for key in collisions} for i in range(n)]
    return list(collisions)


def _instance_attr(sample: dict[str, Any], instance_idx: int, key: str) -> Any | None:
    instances = sample.get("instances", {})
    values = instances.get(key)
    if values is None or instance_idx >= len(values):
        return None
    value = values[instance_idx]
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if key in LABEL_NAMES:
        label_id = int(value)
        names = LABEL_NAMES[key]
        if 0 <= label_id < len(names):
            return names[label_id]
    return value


def _instance_label(sample: dict[str, Any], instance_idx: int) -> str:
    """Return a readable Kubric object label like 'small red metal cube'."""
    parts = [
        _instance_attr(sample, instance_idx, "size_label"),
        _instance_attr(sample, instance_idx, "color_label"),
        _instance_attr(sample, instance_idx, "material_label"),
        _instance_attr(sample, instance_idx, "shape_label"),
    ]
    return " ".join(str(p) for p in parts if p is not None)


def _instance_metadata(sample: dict[str, Any], instance_idx: int) -> dict[str, Any]:
    return {
        "instance_id": instance_idx,
        "label": _instance_label(sample, instance_idx),
        "size_label": _instance_attr(sample, instance_idx, "size_label"),
        "color_label": _instance_attr(sample, instance_idx, "color_label"),
        "material_label": _instance_attr(sample, instance_idx, "material_label"),
        "shape_label": _instance_attr(sample, instance_idx, "shape_label"),
        "color_rgb": (
            np.asarray(_instance_attr(sample, instance_idx, "color"))
            .astype(float)
            .tolist()
            if _instance_attr(sample, instance_idx, "color") is not None
            else None
        ),
    }


def _slug(value: str) -> str:
    return "_".join(value.lower().replace("-", " ").split())


def _frame_path(video_frames_dir: Path, frame_idx: int) -> Path:
    return video_frames_dir / f"frame_{frame_idx:06d}.jpg"


def _depth_path(video_masks_dir: Path, frame_idx: int) -> Path:
    return video_masks_dir / f"frame_{frame_idx:06d}_depth.png"


def _save_frame_if_needed(video: np.ndarray, frame_idx: int, frame_path: Path) -> None:
    if not frame_path.exists():
        Image.fromarray(video[frame_idx]).save(frame_path, quality=95)


def _save_depth_if_needed(depths: np.ndarray | None, frame_idx: int, depth_path: Path) -> str | None:
    if depths is None:
        return None
    if not depth_path.exists():
        _depth_image(depths[frame_idx]).save(depth_path)
    return str(depth_path.resolve())


def _record_no_touch_candidate(
    candidates: dict[int, dict[str, Any]],
    frame_idx: int,
    source: str,
    touch_frame: int | None = None,
    offset: int | None = None,
) -> None:
    entry = candidates.setdefault(
        frame_idx,
        {"sources": set(), "hard_negative_offsets": []},
    )
    entry["sources"].add(source)
    if source == "hard_negative" and touch_frame is not None and offset is not None:
        entry["hard_negative_offsets"].append(
            {"touch_frame": int(touch_frame), "offset": int(offset)}
        )


def convert(
    data_dir: Path,
    dataset: str,
    split: str,
    output_root: Path,
    max_videos: int | None,
    touch_radius: int,
    no_touch_fps: float,
    hard_negative_window: int,
    video_fps: float,
    include_floor_collisions: bool,
) -> None:
    tfds = _require_tfds()

    frames_root = output_root / "frames"
    masks_root = output_root / "masks"
    annotations_root = output_root / "annotations"
    frames_root.mkdir(parents=True, exist_ok=True)
    masks_root.mkdir(parents=True, exist_ok=True)
    annotations_root.mkdir(parents=True, exist_ok=True)

    ds = tfds.load(
        dataset,
        split=split,
        data_dir=str(data_dir),
        shuffle_files=False,
        download=False,
    )

    entries: list[dict[str, Any]] = []
    videos_seen = 0
    skipped_floor = 0
    skipped_empty_masks = 0
    skipped_occluded_touches = 0
    skipped_bad_frame = 0
    skipped_multi_pair_frames = 0
    collapsed_duplicate_contact_events = 0
    depth_frames_written: set[Path] = set()
    videos_without_depth = 0
    no_touch_rows = 0

    iterator = tfds.as_numpy(ds)
    if max_videos is not None:
        iterator = zip(range(max_videos), iterator)
    else:
        iterator = enumerate(iterator)

    for _, sample in tqdm(iterator, desc=f"Converting {dataset}:{split}", unit="video"):
        metadata = sample["metadata"]
        raw_video_name = _decode_text(metadata.get("video_name", f"{split}_{videos_seen:06d}"))
        video_id = Path(raw_video_name).stem or f"{split}_{videos_seen:06d}"

        video = sample["video"]
        segmentations = sample["segmentations"][..., 0]
        depths = sample.get("depth")
        if depths is None:
            videos_without_depth += 1
        num_frames, height, width = video.shape[:3]

        video_frames_dir = frames_root / video_id
        video_masks_dir = masks_root / video_id
        video_frames_dir.mkdir(parents=True, exist_ok=True)
        video_masks_dir.mkdir(parents=True, exist_ok=True)

        collision_events: list[tuple[dict[str, Any], int, int, int]] = []
        collision_frames: set[int] = set()

        for event in _iter_collision_events(sample):
            frame_idx = _as_int(event["frame"])
            if frame_idx < 0 or frame_idx >= num_frames:
                skipped_bad_frame += 1
                continue

            inst_a, inst_b = [int(x) for x in np.asarray(event["instances"]).tolist()]
            if FLOOR_OR_BACKGROUND_INSTANCE in (inst_a, inst_b):
                skipped_floor += 1
                continue
            collision_events.append((event, frame_idx, inst_a, inst_b))
            collision_frames.add(frame_idx)

        no_touch_candidates: dict[int, dict[str, Any]] = {}
        if no_touch_fps > 0:
            sample_every = max(1, round(video_fps / no_touch_fps))
            for frame_idx in range(0, num_frames, sample_every):
                if frame_idx not in collision_frames:
                    _record_no_touch_candidate(no_touch_candidates, frame_idx, "uniform")

        candidates_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for event, frame_idx, inst_a, inst_b in collision_events:
            # Kubric segmentation IDs are one greater than instance indices.
            object1_mask_arr = segmentations[frame_idx] == (inst_a + 1)
            object2_mask_arr = segmentations[frame_idx] == (inst_b + 1)
            if not object1_mask_arr.any() or not object2_mask_arr.any():
                skipped_empty_masks += 1
                continue

            visible_touch_mask = _visible_touch_mask(
                event["image_position"],
                height,
                width,
                touch_radius,
                object1_mask_arr,
                object2_mask_arr,
            )
            if visible_touch_mask is None:
                skipped_occluded_touches += 1
                continue

            candidates_by_frame[frame_idx].append(
                {
                    "event": event,
                    "frame_idx": frame_idx,
                    "inst_a": inst_a,
                    "inst_b": inst_b,
                    "pair_key": tuple(sorted((inst_a, inst_b))),
                    "force": float(np.asarray(event["force"]).item()),
                    "object1_mask_arr": object1_mask_arr,
                    "object2_mask_arr": object2_mask_arr,
                    "visible_touch_mask": visible_touch_mask,
                }
            )

        if hard_negative_window > 0:
            for frame_idx in candidates_by_frame:
                for offset in range(-hard_negative_window, hard_negative_window + 1):
                    if offset == 0:
                        continue
                    negative_frame_idx = frame_idx + offset
                    if 0 <= negative_frame_idx < num_frames and negative_frame_idx not in collision_frames:
                        _record_no_touch_candidate(
                            no_touch_candidates,
                            negative_frame_idx,
                            "hard_negative",
                            touch_frame=frame_idx,
                            offset=offset,
                        )

        for frame_idx, candidates in sorted(candidates_by_frame.items()):
            pair_keys = {candidate["pair_key"] for candidate in candidates}
            if len(pair_keys) > 1:
                skipped_multi_pair_frames += 1
                continue

            collapsed_duplicate_contact_events += len(candidates) - 1
            selected = max(candidates, key=lambda candidate: candidate["force"])
            event = selected["event"]
            inst_a = selected["inst_a"]
            inst_b = selected["inst_b"]
            object1_mask_arr = selected["object1_mask_arr"]
            object2_mask_arr = selected["object2_mask_arr"]
            visible_touch_mask = selected["visible_touch_mask"]

            frame_stem = f"frame_{frame_idx:06d}"
            frame_path = _frame_path(video_frames_dir, frame_idx)
            depth_path = _depth_path(video_masks_dir, frame_idx)
            object1_path = video_masks_dir / f"{frame_stem}_object1.png"
            object2_path = video_masks_dir / f"{frame_stem}_object2.png"
            touch_path = video_masks_dir / f"{frame_stem}_touch.png"

            _save_frame_if_needed(video, frame_idx, frame_path)
            depth_path_str = _save_depth_if_needed(depths, frame_idx, depth_path)
            if depth_path_str is not None:
                depth_frames_written.add(depth_path)
            _binary_mask(object1_mask_arr).save(object1_path)
            _binary_mask(object2_mask_arr).save(object2_path)
            _binary_mask(visible_touch_mask).save(touch_path)

            instance_a = _instance_metadata(sample, inst_a)
            instance_b = _instance_metadata(sample, inst_b)
            object1_label = instance_a["label"]
            object2_label = instance_b["label"]
            object_name = f"{_slug(object1_label)}__{_slug(object2_label)}"
            entries.append(
                {
                    "image_path": str(frame_path.resolve()),
                    "object1_mask_path": str(object1_path.resolve()),
                    "object2_mask_path": str(object2_path.resolve()),
                    "target_path": str(touch_path.resolve()),
                    **({"depth_path": depth_path_str} if depth_path_str is not None else {}),
                    "type": "touch",
                    "video_id": video_id,
                    "object_name": object_name,
                    "object1_label": object1_label,
                    "object2_label": object2_label,
                    "object1_role": "colliding_object_1",
                    "object2_role": "colliding_object_2",
                    "kubric": {
                        "dataset": dataset,
                        "split": split,
                        "frame": frame_idx,
                        "instances": [inst_a, inst_b],
                        "instance_a": instance_a,
                        "instance_b": instance_b,
                        "force": selected["force"],
                        "raw_contact_events_in_frame": len(candidates),
                        "position": np.asarray(event["position"]).astype(float).tolist(),
                        "image_position": np.asarray(event["image_position"]).astype(float).tolist(),
                        "contact_normal": np.asarray(event["contact_normal"]).astype(float).tolist(),
                    },
                }
            )

        for frame_idx, candidate in sorted(no_touch_candidates.items()):
            frame_path = _frame_path(video_frames_dir, frame_idx)
            depth_path = _depth_path(video_masks_dir, frame_idx)
            _save_frame_if_needed(video, frame_idx, frame_path)
            depth_path_str = _save_depth_if_needed(depths, frame_idx, depth_path)
            if depth_path_str is not None:
                depth_frames_written.add(depth_path)
            sources = sorted(candidate["sources"], key=lambda s: (s != "uniform", s))
            offsets = sorted(
                candidate["hard_negative_offsets"],
                key=lambda item: (item["touch_frame"], item["offset"]),
            )
            entries.append(
                {
                    "image_path": str(frame_path.resolve()),
                    **({"depth_path": depth_path_str} if depth_path_str is not None else {}),
                    "type": "no-touch",
                    "video_id": video_id,
                    "object_name": "",
                    "kubric": {
                        "dataset": dataset,
                        "split": split,
                        "frame": frame_idx,
                        "no_touch_sources": sources,
                        "hard_negative_offsets": offsets,
                    },
                }
            )
            no_touch_rows += 1

        videos_seen += 1

    split_name = "val" if split in {"validation", "val"} else split
    annotations_path = annotations_root / f"{split_name}.json"
    ctx_index_path = annotations_root / f"{split_name}_ctx_index.json"
    annotations_path.write_text(json.dumps(entries, indent=2))
    ctx_index_path.write_text(json.dumps(_build_ctx_index(entries)))

    n_touch = sum(1 for entry in entries if entry["type"] == "touch")
    print(f"\nWrote {len(entries)} rows from {videos_seen} video(s).")
    print(f"  touch rows:    {n_touch}")
    print(f"  no-touch rows: {no_touch_rows}")
    print(f"  frames:      {frames_root}")
    print(f"  masks:       {masks_root}")
    print(f"  depth masks: {len(depth_frames_written)}")
    print(f"  annotations: {annotations_path}")
    print(f"  ctx index:   {ctx_index_path}")
    if videos_without_depth:
        print(f"  videos without depth field: {videos_without_depth}")
    if skipped_floor:
        print(f"  skipped floor/background collisions: {skipped_floor}")
    if skipped_empty_masks:
        print(f"  skipped collisions with invisible foreground mask(s): {skipped_empty_masks}")
    if skipped_occluded_touches:
        print(f"  skipped occluded/non-visible touches: {skipped_occluded_touches}")
    if skipped_multi_pair_frames:
        print(f"  skipped frames with multiple object-pair contacts: {skipped_multi_pair_frames}")
    if collapsed_duplicate_contact_events:
        print(f"  collapsed duplicate same-pair contact events: {collapsed_duplicate_contact_events}")
    if skipped_bad_frame:
        print(f"  skipped collisions with invalid frame index: {skipped_bad_frame}")


def main() -> None:
    project_root = _project_root()
    parser = argparse.ArgumentParser(
        description="Convert local Kubric MOVi TFDS files to touch_from_segmentation annotations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=project_root / "data" / "kubric_tfds",
        help="TFDS data_dir containing movi_a/128x128/1.0.0",
    )
    parser.add_argument("--dataset", default="movi_a/128x128")
    parser.add_argument("--split", default="validation")
    parser.add_argument(
        "--output_root",
        type=Path,
        default=project_root / "data" / "kubric_movi_a",
    )
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument(
        "--touch_radius",
        type=int,
        default=3,
        help="Radius in pixels for the binary touch disk around each collision point.",
    )
    parser.add_argument(
        "--no_touch_fps",
        type=float,
        default=2.0,
        help="Uniform no-touch frame sampling rate. Use 0 to disable uniform no-touch sampling.",
    )
    parser.add_argument(
        "--hard_negative_window",
        type=int,
        default=4,
        help="Number of collision-free neighboring frames to sample before and after each touch.",
    )
    parser.add_argument(
        "--video_fps",
        type=float,
        default=12.0,
        help="FPS used to convert --no_touch_fps into a frame stride.",
    )
    parser.add_argument(
        "--include_floor_collisions",
        action="store_true",
        help=(
            "Reserved for experimentation. The current training format needs two "
            "foreground masks, so floor/background collisions are still skipped."
        ),
    )
    args = parser.parse_args()

    convert(
        data_dir=args.data_dir,
        dataset=args.dataset,
        split=args.split,
        output_root=args.output_root,
        max_videos=args.max_videos,
        touch_radius=args.touch_radius,
        no_touch_fps=args.no_touch_fps,
        hard_negative_window=args.hard_negative_window,
        video_fps=args.video_fps,
        include_floor_collisions=args.include_floor_collisions,
    )


if __name__ == "__main__":
    main()
