import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .constants import ANNO_H, ANNO_W, TOUCH_DILATION_RADIUS
from .failure_log import FailureLog

# One lock per video dir to serialise pair_metadata.json updates across threads.
_meta_locks: dict[Path, threading.Lock] = {}
_meta_locks_lock = threading.Lock()


def _meta_lock_for(mask_dir: Path) -> threading.Lock:
    with _meta_locks_lock:
        if mask_dir not in _meta_locks:
            _meta_locks[mask_dir] = threading.Lock()
        return _meta_locks[mask_dir]


def _write_pair_metadata(mask_dir: Path, stem: str, pair_idx: int, rec: dict) -> None:
    """Append one entry to the video-level pair_metadata.json sidecar."""
    meta_path = mask_dir / "pair_metadata.json"
    pair_key  = f"{stem}_p{pair_idx}"
    entry     = {
        "object_name": rec.get("object_name", ""),
        "label":       rec.get("label", ""),
    }
    lock = _meta_lock_for(mask_dir)
    with lock:
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        meta[pair_key] = entry
        meta_path.write_text(json.dumps(meta, indent=2))


def _polygons_to_mask(segments: list, h: int, w: int) -> np.ndarray:
    """Rasterise a list of polygon segment lists into a binary uint8 mask (0/255)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in segments:
        if len(poly) < 3:
            continue
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask


def _scale_segments(segments: list, sx: float, sy: float) -> list:
    if sx == 1.0 and sy == 1.0:
        return segments
    return [[[pt[0] * sx, pt[1] * sy] for pt in poly] for poly in segments]


def _compute_touch(hand_mask: np.ndarray, obj_mask: np.ndarray,
                   radius: int = TOUCH_DILATION_RADIUS) -> np.ndarray:
    """Dilate both masks by radius then intersect to produce the contact zone."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    return cv2.bitwise_and(cv2.dilate(hand_mask, kernel), cv2.dilate(obj_mask, kernel))


def _mask_tag(pair_idx: int) -> str:
    """File-name tag for a contact pair: '_p0', '_p1', …"""
    return f"_p{pair_idx}"


def _render_masks(rec: dict, frames_root: Path, masks_root: Path) -> bool:
    """
    Render hand, object, and touch masks for one frame record.

    VISOR polygon coordinates are in 1920×1080 space; coordinates are scaled
    to the actual downloaded frame dimensions before rasterisation.
    Mask files are named ``{stem}_p{pair_idx}_{hand|object|touch}.png`` so
    that frames with multiple contact pairs each get their own triplet.
    Returns True on success, False if the frame file is missing.
    """
    vid      = rec["video_id"]
    name     = rec["frame_name"]
    stem     = Path(name).stem
    tag      = _mask_tag(rec.get("pair_idx", 0))

    frame_path = frames_root / vid / name
    if not frame_path.exists() or frame_path.stat().st_size == 0:
        return False, "frame file missing or empty"

    try:
        with Image.open(frame_path) as img:
            img.verify()  # catches truncated / corrupt files before we rasterise
        with Image.open(frame_path) as img:
            w, h = img.size  # PIL returns (width, height)
    except Exception as exc:
        return False, f"frame unreadable: {exc}"

    sx, sy = w / ANNO_W, h / ANNO_H
    hand_segs = _scale_segments(rec["hand_segments"], sx, sy)
    obj_segs  = _scale_segments(
        rec["obj_segments"] + rec.get("extra_obj_segs", []), sx, sy
    )

    hand_mask  = _polygons_to_mask(hand_segs, h, w)
    obj_mask   = _polygons_to_mask(obj_segs,  h, w)
    touch_mask = _compute_touch(hand_mask, obj_mask)

    mask_dir = masks_root / vid
    mask_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(hand_mask ).save(mask_dir / f"{stem}{tag}_hand.png")
    Image.fromarray(obj_mask  ).save(mask_dir / f"{stem}{tag}_object.png")
    Image.fromarray(touch_mask).save(mask_dir / f"{stem}{tag}_touch.png")
    _write_pair_metadata(mask_dir, stem, rec.get("pair_idx", 0), rec)
    return True, None


def render_all_masks(sampled: list, frames_root: Path, masks_root: Path,
                     workers: int, flog: FailureLog):
    """Render masks for every record in *sampled* using a thread pool."""
    total = len(sampled)
    ok = fail = skip = 0
    print(f"\n  Rendering masks for {total} frames …")

    def _render(rec):
        stem     = Path(rec["frame_name"]).stem
        tag      = _mask_tag(rec.get("pair_idx", 0))
        mask_dir = masks_root / rec["video_id"]
        if all((mask_dir / f"{stem}{tag}_{t}.png").exists()
               for t in ("hand", "object", "touch")):
            return "skip", None
        try:
            success, detail = _render_masks(rec, frames_root, masks_root)
        except Exception as exc:
            success, detail = False, f"unexpected error: {exc}"
        if not success:
            flog.record("mask", rec["video_id"], rec["frame_name"], detail)
        return ("ok" if success else "fail"), detail

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_render, rec): rec for rec in sampled}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                r, _ = fut.result()
            except Exception as exc:
                rec = futs[fut]
                flog.record("mask", rec["video_id"], rec["frame_name"],
                            f"unexpected error: {exc}")
                r = "fail"
            if r == "ok":     ok   += 1
            elif r == "fail": fail += 1
            else:             skip += 1
            if i % 50 == 0 or i == total:
                print(f"    {i}/{total}  ✓{ok}  skip{skip}  ✗{fail}", end="\r")

    print(f"\n  Done: {ok} rendered, {skip} skipped, {fail} failed")
    if fail:
        print(f"  ⚠ {fail} masks failed — see failure log for details")
