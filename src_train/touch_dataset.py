"""
Dataset for SegGPT fine-tuning on touch-region segmentation.

Pipeline per sample
-------------------
1. Load context and query frames, each with hand, object, and touch masks.
   Context must have non-empty hand and object masks; touch may be empty in
   either frame (a no-touch frame is a valid negative training example).
2. Apply synchronized spatial augmentation (RandomResizedCrop + HFlip) using
   torchvision.transforms.v2 with tv_tensors.Mask so all masks always use
   nearest-neighbour interpolation while images use bicubic.
3. Apply colour jitter to images only (p=0.8 at train time).
4. Sample one shared colour pair (color_A, color_B) for hand and object.
   Both context and query use the same three-class format:
     hand→color_A, object→color_B, touch→_TOUCH_COLOR (fixed bright red).
   When a frame has no touch, the mask contains no red pixels — the format
   is consistent regardless.  Shared hand/object colours activate SegGPT's
   in-context mechanism; the fixed touch colour gives the model a stable
   "red = touch" target with no ambiguity.
5. Include ctx_touch_bin and qry_touch_bin (binary tensors) for per-pixel
   loss upweighting in both halves, and class_colors_norm for nearest-colour
   metric decoding.
6. Manually normalise coloured masks with ImageNet stats, bypassing the HF
   processor's mask_to_rgb() which would destroy the colour information.
7. Pass images through the HF processor for resize + normalisation.
"""

import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.v2 as T
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from transformers import SegGptImageProcessor

sys.path.append(str(Path(__file__).parent))

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# Fixed colour reserved exclusively for the touch class in query target masks.
# It is NEVER used in context masks, so the model learns "this colour = touch"
# purely from the query supervision signal.  Bright red is visually distinctive
# and uncommon as a segmentation class colour for hand or object regions.
_TOUCH_COLOR: tuple[int, int, int] = (255, 0, 0)


def _sample_color_excluding(
    excluded: tuple[int, int, int],
) -> tuple[int, int, int]:
    """Sample a random uint8 RGB colour, retrying if it equals `excluded`."""
    while True:
        color = tuple(int(c) for c in np.random.randint(0, 256, 3))
        if color != excluded:
            return color  # type: ignore[return-value]


def _contactor_mask_path(s: dict) -> str:
    """Return the contacting-agent mask path.

    EK uses 'hand_mask_path'; GH uses 'stick_mask_path'.
    Returns an empty string if neither key is present.
    """
    return s.get("hand_mask_path") or s.get("stick_mask_path") or ""


def _valid_context(s: dict) -> bool:
    """Return True if a sample can serve as a context frame (hand + object non-empty)."""
    contactor = _contactor_mask_path(s)
    return (
        bool(contactor)
        and "object_mask_path" in s
        and bool(np.any(np.array(Image.open(contactor).convert("L")) > 0))
        and bool(np.any(np.array(Image.open(s["object_mask_path"]).convert("L")) > 0))
    )


def _context_key(s: dict) -> str:
    """Return the grouping key used for context sampling.

    If a sample carries ``object_name`` (EPIC Kitchen), the key is the object
    name alone so context is drawn from any video containing that object type,
    maximising sample diversity.

    For datasets without ``object_name`` (Greatest Hits, Manual), each video
    contains only one object type, so ``video_id`` alone is sufficient.
    """
    raw = s.get("object_name")
    # Guard against NaN (float) from the pandas DataFrame→dict roundtrip in
    # generate_annotations.py when some entries lacked object_name.
    object_name = raw.strip() if isinstance(raw, str) else ""
    if object_name:
        return object_name
    return s.get("video_id", "")


def _batch_category(s: dict) -> str:
    """Semantic category label for per-class metric reporting.

    Priority: object_name (EPIC Kitchen) → material (Greatest Hits) → video_id.
    Independent of _context_key so this never affects context sampling logic.
    """
    raw = s.get("object_name")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    mat = s.get("material")
    if isinstance(mat, str) and mat.strip():
        return mat.strip()
    return s.get("video_id", "(unknown)")


def build_context_index(samples: list[dict]) -> dict[str, list[int]]:
    """Build a context lookup table: key → [valid context sample indices].

    Uses :func:`_context_key` per sample, so the grouping is automatically
    object-level for EPIC Kitchen and video-level for other datasets.
    Validity check (:func:`_valid_context`) is run once here to avoid
    repeated file I/O during training.
    """
    index: dict[str, list[int]] = defaultdict(list)
    for idx, s in enumerate(samples):
        try:
            if _valid_context(s):
                index[_context_key(s)].append(idx)
        except OSError:
            pass
    return index


def _normalize_colored_mask(mask_rgb: Image.Image) -> torch.Tensor:
    """Resize to 448×448 and apply ImageNet normalisation to a coloured RGB mask.

    This bypasses SegGptImageProcessor.mask_to_rgb() which would collapse the
    colour channels and destroy the random colour we just painted.
    """
    mask_rgb = mask_rgb.resize((448, 448), Image.NEAREST)
    arr = np.array(mask_rgb).astype(np.float32) / 255.0  # (H, W, 3) in [0,1]
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD  # ImageNet normalise
    return torch.from_numpy(arr).permute(2, 0, 1).float()  # (3, H, W)


def _normalize_color(color: tuple[int, int, int]) -> list[float]:
    """ImageNet-normalise a single (R, G, B) uint8 colour tuple → list of 3 floats."""
    return [
        (color[0] / 255.0 - _IMAGENET_MEAN[0]) / _IMAGENET_STD[0],
        (color[1] / 255.0 - _IMAGENET_MEAN[1]) / _IMAGENET_STD[1],
        (color[2] / 255.0 - _IMAGENET_MEAN[2]) / _IMAGENET_STD[2],
    ]


class TouchPairDataset(Dataset):
    """Yields (context, query) pairs for SegGPT in-context fine-tuning.

    Accepts one or more JSON annotation files produced by generate_annotations.py
    (fields: image_path, hand_mask_path, object_mask_path, target_path, video_id).
    Samples from all files are concatenated; context is always drawn from the same
    object class so the visual prompt stays semantically coherent.

    Context carries a two-class mask (hand + object, random colours per sample).
    Query target carries a three-class mask: the same hand/object colours PLUS
    the fixed ``_TOUCH_COLOR`` for touch pixels.  Touch never appears in the
    context, so the model must infer it from spatial reasoning — exactly what we
    want at inference.  Negative query samples (empty touch mask) are included
    so the model also learns to output "no touch."
    Only samples with non-empty hand and object masks are eligible as context;
    the precomputed ``_ctx_index.json`` files remain valid.
    """

    _color_jitter = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)

    def __init__(
        self,
        json_paths: str | Path | list[str | Path],
        processor: SegGptImageProcessor,
        augment: bool = True,
        object_filter: list[str] | None = None,
        touch_suffix: str | None = None,
    ):
        if isinstance(json_paths, (str, Path)):
            json_paths = [json_paths]

        self.samples: list[dict] = []
        # (local_ctx_index, file_sample_count) per JSON — used to offset indices
        _file_meta: list[tuple[dict[str, list[int]] | None, int]] = []

        for path in json_paths:
            path = Path(path)
            with open(path) as f:
                loaded = json.load(f)
            self.samples.extend(loaded)
            print(f"  Loaded {len(loaded):,} samples from {path}")

            ctx_path = path.with_name(path.stem + "_ctx_index.json")
            local_idx = json.loads(ctx_path.read_text()) if ctx_path.exists() else None
            _file_meta.append((local_idx, len(loaded)))

        if not self.samples:
            raise ValueError(f"No samples found in {json_paths}")

        self.processor = processor
        self.augment = augment
        self.touch_suffix = touch_suffix  # e.g. "refined", "dilated_r20", "refined_dilated_r20"

        # Build context index: key → [valid context sample indices].
        # Prefer the precomputed index (no mask file I/O); fall back to slow build.
        if all(idx is not None for idx, _ in _file_meta):
            merged: dict[str, list[int]] = defaultdict(list)
            offset = 0
            for local_idx, n in _file_meta:
                for key, indices in local_idx.items():
                    merged[key].extend(i + offset for i in indices)
                offset += n
            self._ctx_index = dict(merged)
            print(f"  Context index loaded: {len(self._ctx_index)} groups")
        else:
            missing = [str(p) for p, (idx, _) in zip(json_paths, _file_meta) if idx is None]
            print(f"  No precomputed context index for: {missing}")
            print(f"  Building context index (slow — run generate_annotations.py to precompute)...")
            self._ctx_index = build_context_index(self.samples)

        if object_filter is not None:
            if object_filter == ["auto"]:
                chosen_keys = {max(self._ctx_index, key=lambda k: len(self._ctx_index[k]))}
                print(f"  Auto-selected class: '{next(iter(chosen_keys))}'")
            else:
                invalid = [k for k in object_filter if k not in self._ctx_index]
                if invalid:
                    print(f"  ⚠ {len(invalid)} filter keys not in this dataset — skipping them")
                chosen_keys = set(object_filter) - set(invalid)
                if not chosen_keys:
                    raise ValueError(
                        "No filter keys matched the context index. "
                        f"Available keys: {sorted(self._ctx_index)}"
                    )

            old_valid_sets = {k: set(self._ctx_index.get(k, [])) for k in chosen_keys}
            sorted_keep = sorted(
                i for i, s in enumerate(self.samples) if _context_key(s) in chosen_keys
            )
            new_ctx: dict[str, list[int]] = defaultdict(list)
            for new_i, orig_i in enumerate(sorted_keep):
                key = _context_key(self.samples[orig_i])
                if orig_i in old_valid_sets[key]:
                    new_ctx[key].append(new_i)
            self.samples = [self.samples[i] for i in sorted_keep]
            self._ctx_index = dict(new_ctx)
            total_valid = sum(len(v) for v in new_ctx.values())
            print(
                f"  Class filter {sorted(chosen_keys)}: "
                f"{len(self.samples):,} samples, {total_valid:,} valid contexts"
            )

        self.spatial_transforms = (
            T.Compose(
                [
                    T.RandomResizedCrop(
                        448, scale=(0.3, 1.0), interpolation=T.InterpolationMode.BICUBIC
                    ),
                    T.RandomHorizontalFlip(),
                ]
            )
            if augment
            else T.Resize((448, 448), interpolation=T.InterpolationMode.BICUBIC)
        )

    @staticmethod
    def _open_image(path: str) -> Image.Image:
        """Open an image file with retry logic for transient NFS/cluster I/O errors."""
        while True:
            try:
                return Image.open(path)
            except OSError as e:
                print(f"Caught exception: {e}. Re-trying...")
                time.sleep(1)

    def _load(
        self, idx: int
    ) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image]:
        """Return (image, hand_mask, object_mask, touch_mask) for sample idx."""
        s = self.samples[idx]
        image = self._open_image(s["image_path"]).convert("RGB")
        hand_mask = self._open_image(_contactor_mask_path(s)).convert("L")
        object_mask = self._open_image(s["object_mask_path"]).convert("L")
        target = Path(s["target_path"])
        if self.touch_suffix:
            variant = target.parent / (target.stem + "_" + self.touch_suffix + target.suffix)
            if variant.exists():
                target = variant
        touch_mask = self._open_image(str(target)).convert("L")
        return image, hand_mask, object_mask, touch_mask

    def _apply_spatial(self, image: Image.Image, *masks: Image.Image) -> tuple:
        """Synchronized spatial transform for an image and one or more masks.

        tv_tensors.Mask causes transforms.v2 to use NEAREST interpolation for
        masks automatically, while the image gets BICUBIC.
        """
        img_tv = tv_tensors.Image(F.to_image(image))
        masks_tv = [tv_tensors.Mask(F.to_image(m)) for m in masks]
        results = self.spatial_transforms(img_tv, *masks_tv)
        img_out = F.to_pil_image(results[0])
        masks_out = tuple(F.to_pil_image(r) for r in results[1:])
        return (img_out,) + masks_out

    def _maybe_color_jitter(self, image: Image.Image) -> Image.Image:
        if self.augment and torch.rand(1).item() < 0.8:
            return self._color_jitter(image)
        return image

    @staticmethod
    def _dual_color_mask(
        hand_mask: Image.Image,
        object_mask: Image.Image,
        color_a: tuple[int, int, int],
        color_b: tuple[int, int, int],
    ) -> Image.Image:
        """Two-class RGB mask using pre-sampled colours: hand → color_A, object → color_B."""
        arr_h = np.array(hand_mask)
        arr_o = np.array(object_mask)
        rgb = np.zeros((*arr_h.shape, 3), dtype=np.uint8)
        rgb[arr_h > 0] = color_a
        rgb[arr_o > 0] = color_b
        return Image.fromarray(rgb, mode="RGB")

    @staticmethod
    def _single_color_mask(mask: Image.Image) -> Image.Image:
        """Single-class RGB mask: positive pixels → random color, background black."""
        color = tuple(int(c) for c in np.random.randint(0, 256, 3))
        arr = np.array(mask)
        rgb = np.zeros((*arr.shape, 3), dtype=np.uint8)
        rgb[arr > 0] = color
        return Image.fromarray(rgb, mode="RGB")

    @staticmethod
    def _triple_color_mask(
        hand_mask: Image.Image,
        object_mask: Image.Image,
        touch_mask: Image.Image,
        color_a: tuple[int, int, int],
        color_b: tuple[int, int, int],
        color_c: tuple[int, int, int],
    ) -> Image.Image:
        """Three-class RGB mask using pre-sampled shared colours.

        hand → color_A, object → color_B, touch → color_C (painted last so it
        overrides any overlap with hand/object).  Colours are sampled once per
        __getitem__ call and reused for both context and query so SegGPT's
        in-context mechanism can match classes across the two frames.
        """
        arr_h = np.array(hand_mask)
        arr_o = np.array(object_mask)
        arr_t = np.array(touch_mask)
        rgb = np.zeros((*arr_h.shape, 3), dtype=np.uint8)
        rgb[arr_h > 0] = color_a
        rgb[arr_o > 0] = color_b
        rgb[arr_t > 0] = color_c  # touch on top — overrides hand/object
        return Image.fromarray(rgb, mode="RGB")

    def _pick_context_idx(self, query_idx: int) -> int:
        """Pick a context index sharing the same context key as the query.

        For EPIC Kitchen the key is ``video_id::object_name``, so context is
        always a frame of the same hand-object interaction.  For other datasets
        the key is ``video_id``.
        """
        key        = _context_key(self.samples[query_idx])
        candidates = [i for i in self._ctx_index.get(key, []) if i != query_idx]
        if not candidates:
            candidates = list(self._ctx_index.get(key, []))
        if not candidates:
            raise ValueError(
                f"No valid context candidate for key '{key}'. "
                "Check your annotation files."
            )
        return random.choice(candidates)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ctx_idx = self._pick_context_idx(idx)

        ctx_img, ctx_hand, ctx_obj, ctx_touch = self._load(ctx_idx)
        qry_img, qry_hand, qry_obj, qry_touch = self._load(idx)

        # 1. Synchronized spatial augmentation (same crop/flip for img + masks)
        ctx_img, ctx_hand, ctx_obj, ctx_touch = self._apply_spatial(
            ctx_img, ctx_hand, ctx_obj, ctx_touch
        )
        qry_img, qry_hand, qry_obj, qry_touch = self._apply_spatial(
            qry_img, qry_hand, qry_obj, qry_touch
        )

        # 2. Colour jitter on images only — never on masks
        ctx_img = self._maybe_color_jitter(ctx_img)
        qry_img = self._maybe_color_jitter(qry_img)

        # 3. Shared random colours for hand/object; touch always uses _TOUCH_COLOR
        color_a = _sample_color_excluding(_TOUCH_COLOR)
        color_b = _sample_color_excluding(_TOUCH_COLOR)

        # 4. Both halves use the same three-class format with the fixed touch colour.
        #    When ctx_touch is empty the context mask is effectively two-class (no red
        #    pixels), so there is no format mismatch regardless of whether the context
        #    frame contains touch or not.
        ctx_mask_rgb = self._triple_color_mask(
            ctx_hand, ctx_obj, ctx_touch, color_a, color_b, _TOUCH_COLOR
        )
        qry_mask_rgb = self._triple_color_mask(
            qry_hand, qry_obj, qry_touch, color_a, color_b, _TOUCH_COLOR
        )

        # 5. Images through HF processor (resize 448×448 already done, normalize)
        img_inputs = self.processor(
            images=qry_img,
            prompt_images=ctx_img,
            return_tensors="pt",
        )

        # 6. Binary touch masks — for per-pixel loss upweighting in both halves
        ctx_touch_bin = torch.from_numpy(
            (np.array(ctx_touch) > 0).astype(np.float32)
        )  # (448, 448)
        qry_touch_bin = torch.from_numpy(
            (np.array(qry_touch) > 0).astype(np.float32)
        )  # (448, 448)

        # 7. Normalised colour triplet — for nearest-colour metric decoding
        #    Row 2 is always _normalize_color(_TOUCH_COLOR) (fixed)
        class_colors_norm = torch.tensor(
            [
                _normalize_color(color_a),
                _normalize_color(color_b),
                _normalize_color(_TOUCH_COLOR),
            ],
            dtype=torch.float32,
        )  # (3, 3): rows = [hand_norm, obj_norm, touch_norm]

        return {
            "pixel_values": img_inputs["pixel_values"].squeeze(0),
            "prompt_pixel_values": img_inputs["prompt_pixel_values"].squeeze(0),
            "prompt_masks": _normalize_colored_mask(ctx_mask_rgb),
            "labels": _normalize_colored_mask(qry_mask_rgb),
            "ctx_touch_bin": ctx_touch_bin,
            "qry_touch_bin": qry_touch_bin,
            "class_colors_norm": class_colors_norm,
            "category": _batch_category(self.samples[idx]),
        }
