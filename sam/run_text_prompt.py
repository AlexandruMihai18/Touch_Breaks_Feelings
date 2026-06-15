from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from sam import PromptMaskResult, run_grounded_sam2, run_sam3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run text-prompt segmentation with Grounding DINO + SAM2 and/or SAM3."
    )
    parser.add_argument(
        "--image",
        required=True,
        type=Path,
        help="Path to the RGB image to segment.",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Text prompt for the object of interest, e.g. 'cup' or 'hand'.",
    )
    parser.add_argument(
        "--backend",
        choices=("grounded_sam2", "sam3", "both"),
        default="both",
        help="Which backend to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/sam_text_prompt"),
        help="Directory where masks and debug images will be written.",
    )
    parser.add_argument(
        "--sam3-model-id",
        default=None,
        help="Optional Hugging Face model ID for SAM3. Defaults to facebook/sam3.",
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.3,
        help="Grounding DINO box threshold for the grounded_sam2 backend.",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.25,
        help="Grounding DINO text threshold for the grounded_sam2 backend.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="SAM3 score threshold.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_np = np.array(Image.open(args.image).convert("RGB"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[PromptMaskResult] = []
    if args.backend in {"grounded_sam2", "both"}:
        results.append(
            run_grounded_sam2(
                image_np,
                args.prompt,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
            )
        )
    if args.backend in {"sam3", "both"}:
        results.append(
            run_sam3(
                image_np,
                args.prompt,
                model_id=args.sam3_model_id,
                score_threshold=args.score_threshold,
            )
        )

    for result in results:
        stem = f"{args.image.stem}_{result.backend}"
        mask_path = args.output_dir / f"{stem}_mask.png"
        Image.fromarray(result.mask, mode="L").save(mask_path)

        if result.debug_image is not None:
            debug_path = args.output_dir / f"{stem}_debug.png"
            Image.fromarray(result.debug_image).save(debug_path)

        print(
            f"{result.backend}: prompt={result.prompt!r} "
            f"score={result.score:.4f} box={_round_box(result.box)} mask={mask_path}"
        )


def _round_box(box: list[float]) -> list[float]:
    return [round(v, 2) for v in box]


if __name__ == "__main__":
    main()

