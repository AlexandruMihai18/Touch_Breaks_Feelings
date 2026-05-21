"""Shared constants and data paths used across all tabs."""

import sys
from pathlib import Path

_ROOT = Path(__file__).parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

OUT_H = 400
FRAME_H = 200

FRAMES_ROOT = _ROOT / "data" / "manual_annotations" / "frames"
MASKS_ROOT = _ROOT / "data" / "manual_annotations" / "masks"
MASKS_ROOT.mkdir(parents=True, exist_ok=True)

GH_FRAMES_ROOT = _ROOT / "data" / "greatest_hits" / "frames"
GH_MASKS_ROOT = _ROOT / "data" / "greatest_hits" / "masks"
GH_ANNO_DIR = _ROOT / "data" / "greatest_hits" / "annotations"
GH_MASKS_ROOT.mkdir(parents=True, exist_ok=True)

EK_FRAMES_ROOT    = _ROOT / "data" / "epic_kitchen" / "frames"
EK_MASKS_ROOT     = _ROOT / "data" / "epic_kitchen" / "masks"
EK_ANNO_DIR       = _ROOT / "data" / "epic_kitchen" / "annotations"
EK_COVERAGE_PATH  = _ROOT / "data" / "epic_kitchen" / "object_labels_coverage.txt"
EK_MASKS_ROOT.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "Manual Annotations": (FRAMES_ROOT, MASKS_ROOT),
    "Greatest Hits": (GH_FRAMES_ROOT, GH_MASKS_ROOT),
    "EPIC Kitchen": (EK_FRAMES_ROOT, EK_MASKS_ROOT),
}
DATASET_NAMES = list(DATASETS.keys())
