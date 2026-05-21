from pathlib import Path

from .generate_annotations import generate


def run_generate_annotations(
    output_dir: Path,
    val_frac: float,
    audio_root: Path | None = None,
) -> None:
    generate(
        frames_dir=output_dir / "frames",
        masks_dir=output_dir / "masks",
        output_dir=output_dir / "annotations",
        val_frac=val_frac,
        audio_root=audio_root,
    )
