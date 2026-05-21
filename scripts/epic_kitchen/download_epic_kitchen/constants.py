from pathlib import Path

VISOR_BASE = "https://data.bris.ac.uk/datasets/2v6cgv1x04ol22qp9rm9x2j6a7"
ANNO_BASE  = f"{VISOR_BASE}/GroundTruth-SparseAnnotations/annotations"
FRAME_BASE = f"{VISOR_BASE}/GroundTruth-SparseAnnotations/rgb_frames"

# VISOR polygon coordinates are always in 1920×1080 annotation space.
ANNO_W, ANNO_H = 1920, 1080

MAX_VIDEO_SUFFIX = 130

ALL_PARTICIPANTS = [
    "P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08",
    "P10", "P11", "P12", "P13", "P14", "P15", "P17", "P18",
    "P20", "P22", "P23", "P24", "P25", "P26", "P27", "P28",
    "P30", "P32", "P35", "P37",
]

# in_contact_object values that mean "not in contact" or "indeterminate"
NOT_CONTACT = frozenset([
    "hand-not-in-contact",
    "glove-not-in-contact",
    "none-of-the-above",
    "inconclusive",
])

HAND_NAMES = frozenset(["left hand", "right hand"])
GLOVE_NAMES = frozenset([
    "oven glove", "gloves", "rubber glove",
    "left glove", "right glove", "glove",
])

TOUCH_DILATION_RADIUS = 10

_FILTERED_LABELS_PATH = (
    Path(__file__).parents[3] / "data" / "epic_kitchen" / "object_labels_filtered.txt"
)


def _load_allowed_objects(path: Path) -> frozenset:
    names: set[str] = set()
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                names.add(parts[1])
    return frozenset(names)


# Object names that pass both the surface filter and the coverage threshold.
# Used in Phase 2 to restrict no-contact object candidates to meaningful labels.
ALLOWED_OBJECT_NAMES: frozenset = _load_allowed_objects(_FILTERED_LABELS_PATH)
