"""Lazy model loading for SAM2 and Grounding DINO.

Models are loaded on first use, not at import time. This keeps import cost
near-zero for code that imports detection helpers but hasn't yet processed a
frame, and avoids loading GPU models in scripts that only do CPU work.

Usage
-----
    from touch_detection_alg.greatest_hits.models import get_sam, get_gdino

    sam_processor, sam_model = get_sam()
    gdino_processor, gdino_model = get_gdino()
"""

import torch

INFERENCE_SEED = 42

_sam_processor = None
_sam_model = None
_gdino_processor = None
_gdino_model = None


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_sam():
    """Return (Sam2Processor, Sam2Model), loading on first call."""
    global _sam_processor, _sam_model
    if _sam_model is None:
        from transformers import Sam2Model, Sam2Processor

        _sam_processor = Sam2Processor.from_pretrained("facebook/sam2-hiera-small")
        _sam_model = Sam2Model.from_pretrained("facebook/sam2-hiera-small").to(_device())
        _sam_model.eval()

        torch.manual_seed(INFERENCE_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(INFERENCE_SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    return _sam_processor, _sam_model


def get_gdino():
    """Return (GroundingDinoProcessor, GroundingDinoForObjectDetection), loading on first call."""
    global _gdino_processor, _gdino_model
    if _gdino_model is None:
        from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor

        _ID = "IDEA-Research/grounding-dino-tiny"
        _gdino_processor = GroundingDinoProcessor.from_pretrained(_ID)
        _gdino_model = GroundingDinoForObjectDetection.from_pretrained(_ID).to(_device())
        _gdino_model.eval()

    return _gdino_processor, _gdino_model
