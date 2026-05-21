import json
import random
from collections import defaultdict
from pathlib import Path

from .constants import ALLOWED_OBJECT_NAMES, GLOVE_NAMES, HAND_NAMES, NOT_CONTACT


def _parse_frame(frame: dict, split: str) -> list[dict]:
    """
    Parse one annotated frame and return one record per unique contact pair.

    All hand segments in the frame (both hands) are combined and used in
    every record.  Hands touching the same object UUID are merged into one
    pair; hands touching different objects become separate records.  When no
    contact exists, up to 5 no-contact records are emitted — one per allowed
    object visible in the frame (filtered by ALLOWED_OBJECT_NAMES to exclude
    surfaces and tiny objects).  Falls back to a hand-only record when no
    allowed object is present.

    Each record carries a ``pair_idx`` (0-based, per-frame) used by Phase 4
    to name mask files as ``{stem}_p{i}_hand.png`` etc.
    """
    img  = frame["image"]
    anns = frame["annotations"]
    id_to_ann = {a["id"]: a for a in anns}

    video_id    = img["image_path"].split("/")[0]
    participant = video_id[:3]

    # Collect ALL hand segments from the frame (both hands combined).
    all_hand_segs: list = []

    # Group contact pairs by touched-object UUID.
    pairs: dict[str, dict] = {}

    for ann in anns:
        name = ann.get("name", "")
        if name not in HAND_NAMES and name not in GLOVE_NAMES:
            # skip non-hand annotations
            continue
        # accumulate both hands
        all_hand_segs.extend(ann.get("segments", []))
        ic = ann.get("in_contact_object")
        if ic is None or ic in NOT_CONTACT:
            # hand visible but not touching anything
            continue
        obj_ann = id_to_ann.get(ic)
        if obj_ann is None:
            # referenced object missing from frame
            continue
        if ic not in pairs:
            # deduplicate: two hands on same object → one pair
            pairs[ic] = {
                "obj_segs":    obj_ann.get("segments", []),
                "object_name": obj_ann.get("name", ""),
            }

    # Map object class name → all annotation objects of that class in this frame.
    # Used to collect same-class instances that are not the directly-contacted one.
    name_to_anns_map: dict[str, list] = defaultdict(list)
    for ann in anns:
        ann_name = ann.get("name", "")
        if ann_name not in HAND_NAMES and ann_name not in GLOVE_NAMES and ann.get("segments"):
            # group by class label (e.g. all "cup" instances)
            name_to_anns_map[ann_name].append(ann)

    base = {
        "video_id":    video_id,
        "participant": participant,
        "split":       split,
        "frame_name":  img["name"],
        "image_path":  img["image_path"],
    }

    # One record per unique contacted object UUID.
    # Two hands on the same object → one pair (deduplicated above).
    # Two hands on different objects → two separate pairs → two records.
    # Each record carries ALL hand segments but only the object class it was paired with,
    # so classes are never mixed across records.
    records: list[dict] = []
    for pair_idx, (contact_id, pair) in enumerate(pairs.items()):
        if not pair["obj_segs"]:
            continue
        # Collect every OTHER visible instance of the SAME class as the touched object.
        # The lookup key is pair["object_name"], so only same-class annotations are
        # considered — plates never appear in a cup record, etc.
        extra_obj_segs: list = []
        for ann in name_to_anns_map.get(pair["object_name"], []):
            if ann["id"] != contact_id:
                # skip the directly-touched instance (already in obj_segs)
                extra_obj_segs.extend(ann.get("segments", []))
        records.append({**base, "label": "contact", "pair_idx": pair_idx,
                        "object_name":    pair["object_name"],
                        "hand_segments":  all_hand_segs,
                        "obj_segments":   pair["obj_segs"],        # directly-touched instance
                        "extra_obj_segs": extra_obj_segs})         # other same-class instances; merged in phase 4

    if not records and all_hand_segs:
        # No contact detected — emit up to 5 allowed objects as negative examples.
        obj_candidates = [
            a for a in anns
            if a.get("name", "") not in HAND_NAMES
            and a.get("name", "") not in GLOVE_NAMES
            and a.get("name", "") in ALLOWED_OBJECT_NAMES
            and a.get("segments")
        ][:5]
        if obj_candidates:
            for pair_idx, obj_ann in enumerate(obj_candidates):
                obj_name = obj_ann.get("name", "")
                extra_obj_segs: list = []
                for ann in name_to_anns_map.get(obj_name, []):
                    if ann["id"] != obj_ann["id"]:
                        # skip the candidate itself (already in obj_segs)
                        extra_obj_segs.extend(ann.get("segments", []))
                records.append({**base, "label": "no_contact", "pair_idx": pair_idx,
                                "object_name":    obj_name,
                                "hand_segments":  all_hand_segs,
                                "obj_segments":   obj_ann.get("segments", []),
                                "extra_obj_segs": extra_obj_segs})
        else:
            # Hand present but no allowed object visible at all — hand-only record.
            records.append({**base, "label": "no_contact", "pair_idx": 0,
                            "object_name":    "",
                            "hand_segments":  all_hand_segs,
                            "obj_segments":   [],
                            "extra_obj_segs": []})

    return records


def build_catalog(json_paths: list[Path], split: str) -> dict:
    """
    Parse all annotation JSONs and return:
      {"contact": [...records...], "no_contact": [...records...]}
    """
    catalog: dict[str, list] = {"contact": [], "no_contact": []}

    for jpath in json_paths:
        try:
            data = json.loads(jpath.read_text())
        except json.JSONDecodeError:
            print(f"  ⚠ Bad JSON: {jpath.name}")
            continue

        for frame in data.get("video_annotations", []):
            for rec in _parse_frame(frame, split):
                catalog[rec["label"]].append(rec)

    print(f"\n  Catalog totals:")
    print(f"    contact    : {len(catalog['contact']):,}")
    print(f"    no_contact : {len(catalog['no_contact']):,}")
    return catalog


def stratified_sample(items: list, n: int, strata_keys: list[str], rng) -> list:
    """
    Round-robin sample n items from a list, stratified by the given keys.
    Falls back to returning all items when n >= len(items).
    """
    if n >= len(items):
        print(f"    (requested {n}, only {len(items)} available — taking all)")
        return list(items)

    buckets: dict = defaultdict(list)
    for item in items:
        buckets[tuple(item[k] for k in strata_keys)].append(item)
    for b in buckets.values():
        rng.shuffle(b)

    selected: list = []
    bucket_list = list(buckets.values())
    i = 0
    while len(selected) < n and any(bucket_list):
        b = bucket_list[i % len(bucket_list)]
        if b:
            selected.append(b.pop())
        i += 1
        bucket_list = [b for b in bucket_list if b]

    rng.shuffle(selected)
    return selected


def select_frames(catalog: dict, n_contact: int, n_no_contact: int,
                  seed: int, take_all: bool) -> list:
    """
    Return the list of frame records to process.
    When take_all=True every catalog entry is returned without sampling.
    """
    if take_all:
        sampled = catalog["contact"] + catalog["no_contact"]
        print(f"\n  --all: taking every matched frame")
        print(f"    contact    : {len(catalog['contact']):,}")
        print(f"    no_contact : {len(catalog['no_contact']):,}")
        print(f"    total      : {len(sampled):,}")
        return sampled

    rng = random.Random(seed)
    c = stratified_sample(catalog["contact"],    n_contact,    ["video_id", "participant"], rng)
    n = stratified_sample(catalog["no_contact"], n_no_contact, ["participant"],              rng)
    sampled = c + n
    print(f"\n  Sampled: {len(c)} contact + {len(n)} no-contact = {len(sampled)} total")
    return sampled
