import torch
from transformers import (GroundingDinoForObjectDetection, GroundingDinoProcessor,
                          Sam2Model, Sam2Processor,
                          SegGptForImageSegmentation, SegGptImageProcessor)

device = "cuda" if torch.cuda.is_available() else "cpu"

INFERENCE_SEED = 42
torch.manual_seed(INFERENCE_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(INFERENCE_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# SAM2
processor = Sam2Processor.from_pretrained("facebook/sam2-hiera-small")
sam_model = Sam2Model.from_pretrained("facebook/sam2-hiera-small").to(device)
sam_model.eval()

# Grounding DINO (text → bounding boxes for SAM)
_GDINO_ID = "IDEA-Research/grounding-dino-tiny"
grounding_processor = GroundingDinoProcessor.from_pretrained(_GDINO_ID)
grounding_model = GroundingDinoForObjectDetection.from_pretrained(_GDINO_ID).to(device)
grounding_model.eval()

# SegGPT
_SEGGPT_ID = "BAAI/seggpt-vit-large"
image_processor = SegGptImageProcessor.from_pretrained(_SEGGPT_ID)
seggpt_model = SegGptForImageSegmentation.from_pretrained(_SEGGPT_ID).to(device)
seggpt_model.eval()
