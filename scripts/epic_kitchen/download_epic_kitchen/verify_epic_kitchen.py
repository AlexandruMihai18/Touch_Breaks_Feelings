"""
Sanity-check the downloaded VISOR data.

Run after `python scripts/epic_kitchen/download_epic_kitchen` to confirm everything is in
order and get a quick preview of the contact annotations you'll be working with.

Usage:
    python -m scripts.epic_kitchen.download_epic_kitchen.verify_epic_kitchen
    python -m scripts.epic_kitchen.download_epic_kitchen.verify_epic_kitchen --data-dir /path/to/epic_kitchen
    python -m scripts.epic_kitchen.download_epic_kitchen.verify_epic_kitchen --split val
"""

import json
import argparse
from pathlib import Path
from collections import Counter

# Mirrors constants.py — kept inline so this module stays self-contained.
_NOT_CONTACT = frozenset([
    "hand-not-in-contact", "glove-not-in-contact",
    "none-of-the-above", "inconclusive",
])
_HAND_NAMES  = frozenset(["left hand", "right hand"])
_GLOVE_NAMES = frozenset(["oven glove", "gloves", "rubber glove",
                           "left glove", "right glove", "glove"])


def verify(
    data_dir: str | Path = "./data/epic_kitchen",
    split: str = "train",
    participant: str | None = None,
) -> bool:
    """Run all verification checks. Returns True if the dataset is ready for training."""
    DATA        = Path(data_dir)
    PARTICIPANT = participant
    SPLIT       = split

    ANNO_DIR       = DATA / "visor" / "GroundTruth-SparseAnnotations" / "annotations" / SPLIT
    FRAMES_ROOT    = DATA / "frames"
    MASKS_ROOT     = DATA / "masks"
    FINAL_ANNO_DIR = DATA / "annotations"
    FAILURE_LOG    = DATA / "failures.json"

    print("=" * 68)
    print(f"  VISOR Data Verification")
    print(f"  data-dir    : {DATA}")
    print(f"  split       : {SPLIT}")
    print(f"  participant : {PARTICIPANT or 'ALL'}")
    print("=" * 68)

    # ── 1. Annotation JSONs ───────────────────────────────────────────────────
    print(f"\n── 1. Annotation JSONs ──")
    if not ANNO_DIR.exists():
        print(f"  ✗ Annotation directory not found: {ANNO_DIR}")
        print(f"  → Run: python scripts/epic_kitchen/download_epic_kitchen --output-dir {DATA}")
        return False

    glob_pat = f"{PARTICIPANT}_*.json" if PARTICIPANT else "*.json"
    json_files = sorted(ANNO_DIR.glob(glob_pat))
    print(f"  ✓ {len(json_files)} annotation JSON files in {ANNO_DIR.relative_to(DATA)}")

    if not json_files:
        print(f"  ✗ No JSONs matched. Check download or --participant value.")
        return False

    # ── 2. Parse contact annotations ─────────────────────────────────────────
    print(f"\n── 2. Contact annotation breakdown ──")

    total_frames        = 0
    contact_frames      = 0
    no_contact_frames   = 0
    other_frames        = 0
    contact_objects: Counter = Counter()
    hand_side_counts: Counter = Counter()
    video_stats: dict  = {}

    for jf in json_files:
        try:
            data = json.loads(jf.read_text())
        except json.JSONDecodeError:
            print(f"  ⚠ Bad JSON: {jf.name}")
            continue

        frame_list = data.get("video_annotations", []) if isinstance(data, dict) else data
        video_id   = jf.stem
        vid_total  = 0
        vid_contact = 0

        for frame in frame_list:
            vid_total   += 1
            total_frames += 1

            anns       = frame.get("annotations", [])
            id_to_ann  = {a["id"]: a for a in anns}
            frame_contact = False

            for ann in anns:
                name = ann.get("name", "")
                if name not in _HAND_NAMES and name not in _GLOVE_NAMES:
                    continue

                ic = ann.get("in_contact_object")
                if ic is None or ic in _NOT_CONTACT:
                    continue

                obj_ann = id_to_ann.get(ic)
                if obj_ann is None:
                    continue

                frame_contact = True
                contact_objects[obj_ann.get("name", "unknown")] += 1
                if "left" in name:
                    hand_side_counts["left"] += 1
                elif "right" in name:
                    hand_side_counts["right"] += 1

            if frame_contact:
                contact_frames += 1
                vid_contact    += 1
            else:
                has_hand = any(
                    a.get("name", "") in _HAND_NAMES or a.get("name", "") in _GLOVE_NAMES
                    for a in anns
                )
                if has_hand:
                    no_contact_frames += 1
                else:
                    other_frames += 1

        video_stats[video_id] = {
            "total":   vid_total,
            "contact": vid_contact,
            "rate":    vid_contact / vid_total if vid_total else 0,
        }

    pct = lambda n: f"({100*n/total_frames:.1f}%)" if total_frames else ""
    print(f"  Total annotated frames :  {total_frames:,}")
    print(f"  Contact frames         :  {contact_frames:,}  {pct(contact_frames)}")
    print(f"  No-contact frames      :  {no_contact_frames:,}  {pct(no_contact_frames)}")
    print(f"  No hand present        :  {other_frames:,}  {pct(other_frames)}")
    print(f"\n  Hand side breakdown:")
    print(f"    Left hand contacts   :  {hand_side_counts['left']:,}")
    print(f"    Right hand contacts  :  {hand_side_counts['right']:,}")

    # ── 3. Top contacted objects ──────────────────────────────────────────────
    print(f"\n── 3. Top 15 contacted objects ──")
    if contact_objects:
        max_count = max(contact_objects.values())
        for obj, count in contact_objects.most_common(15):
            bar = "█" * min(int(count / max_count * 30), 30)
            print(f"  {obj:<30} {count:>5}  {bar}")
    else:
        print("  (none found)")

    # ── 4. Per-video contact rates ────────────────────────────────────────────
    print(f"\n── 4. Per-video contact rates (top 10) ──")
    sorted_vids = sorted(video_stats.items(), key=lambda x: x[1]["contact"], reverse=True)
    print(f"  {'Video':<15} {'Total':>7} {'Contact':>8}  Rate")
    print(f"  {'-'*50}")
    for vid, s in sorted_vids[:10]:
        bar = "█" * int(s["rate"] * 20)
        print(f"  {vid:<15} {s['total']:>7,} {s['contact']:>8,}  {s['rate']:.0%}  {bar}")

    # ── 5. Downloaded frames ──────────────────────────────────────────────────
    print(f"\n── 5. Downloaded frames ({FRAMES_ROOT.relative_to(DATA)}) ──")
    total_jpgs = 0
    if not FRAMES_ROOT.exists():
        print(f"  ✗ frames/ not found — Phase 3 may not have run yet.")
    else:
        video_dirs  = [d for d in FRAMES_ROOT.iterdir() if d.is_dir() and
                       (PARTICIPANT is None or d.name.startswith(PARTICIPANT))]
        total_jpgs  = sum(len(list(d.glob("*.jpg"))) for d in video_dirs)
        if total_jpgs:
            print(f"  ✓ {total_jpgs:,} frames across {len(video_dirs)} video dirs")
            if video_dirs:
                sample_vid = sorted(video_dirs)[0]
                sample_jpgs = list(sample_vid.glob("*.jpg"))
                print(f"    e.g. {sample_vid.name}/  → {len(sample_jpgs)} frames  "
                      f"(first: {sample_jpgs[0].name if sample_jpgs else '—'})")
        else:
            print(f"  ✗ No frames found in {FRAMES_ROOT}")

    # ── 6. Rendered masks ─────────────────────────────────────────────────────
    print(f"\n── 6. Rendered masks ({MASKS_ROOT.relative_to(DATA)}) ──")
    hand_masks = 0
    object_masks = 0
    touch_masks = 0
    if not MASKS_ROOT.exists():
        print(f"  ✗ masks/ not found — Phase 4 may not have run yet.")
    else:
        mask_video_dirs = [d for d in MASKS_ROOT.iterdir() if d.is_dir() and
                           (PARTICIPANT is None or d.name.startswith(PARTICIPANT))]
        hand_masks   = sum(len(list(d.glob("*_hand.png")))   for d in mask_video_dirs)
        object_masks = sum(len(list(d.glob("*_object.png"))) for d in mask_video_dirs)
        touch_masks  = sum(len(list(d.glob("*_touch.png")))  for d in mask_video_dirs)

        if hand_masks:
            print(f"  ✓ {hand_masks:,} hand masks, {object_masks:,} object masks, "
                  f"{touch_masks:,} touch masks")
            if hand_masks == object_masks == touch_masks:
                print(f"  ✓ Mask triplets are balanced")
            else:
                print(f"  ⚠ Mask counts differ — some renders may have failed")

            if FRAMES_ROOT.exists() and total_jpgs:
                sample_vid_dir = sorted(mask_video_dirs)[0] if mask_video_dirs else None
                if sample_vid_dir:
                    sample_masks = list(sample_vid_dir.glob("*_touch.png"))
                    if sample_masks:
                        stem = sample_masks[0].stem.replace("_touch", "")
                        frame_file = FRAMES_ROOT / sample_vid_dir.name / f"{stem}.jpg"
                        if frame_file.exists():
                            print(f"  ✓ Frame↔mask alignment PASSED  ({stem}.jpg)")
                        else:
                            print(f"  ⚠ Frame for mask not found: {frame_file}")
        else:
            print(f"  ✗ No masks found in {MASKS_ROOT}")

    # ── 7. Final annotation JSONs ─────────────────────────────────────────────
    print(f"\n── 7. Final annotation JSONs ({FINAL_ANNO_DIR.relative_to(DATA)}) ──")
    for split_name in ("train", "val"):
        anno_file = FINAL_ANNO_DIR / f"{split_name}.json"
        if anno_file.exists():
            try:
                entries = json.loads(anno_file.read_text())
                n = len(entries) if isinstance(entries, list) else "?"
                print(f"  ✓ {split_name}.json  — {n} entries")
            except json.JSONDecodeError:
                print(f"  ✗ {split_name}.json  — malformed JSON")
        else:
            print(f"  ✗ {split_name}.json  — not found (run without --skip-annotations)")

    # ── 8. Failure log ────────────────────────────────────────────────────────
    print(f"\n── 8. Failure log ──")
    if FAILURE_LOG.exists():
        try:
            fdata  = json.loads(FAILURE_LOG.read_text())
            phases = {}
            for entry in fdata:
                phases[entry.get("phase", "?")] = phases.get(entry.get("phase", "?"), 0) + 1
            total_failures = len(fdata)
            if total_failures == 0:
                print(f"  ✓ failures.json — no failures recorded")
            else:
                print(f"  ⚠ {total_failures} failures recorded:")
                for phase, count in sorted(phases.items()):
                    print(f"    phase={phase}  {count} failures")
        except json.JSONDecodeError:
            print(f"  ✗ failures.json — malformed")
    else:
        print(f"  ✗ failures.json — not found (run download first)")

    # ── 9. Sample contact annotation ─────────────────────────────────────────
    print(f"\n── 9. Sample contact annotation ──")
    sample_contact_frame = None
    for jf in json_files:
        try:
            data = json.loads(jf.read_text())
        except json.JSONDecodeError:
            continue
        frame_list = data.get("video_annotations", []) if isinstance(data, dict) else data
        for frame in frame_list:
            anns      = frame.get("annotations", [])
            id_to_ann = {a["id"]: a for a in anns}
            for ann in anns:
                name = ann.get("name", "")
                if name not in _HAND_NAMES and name not in _GLOVE_NAMES:
                    continue
                ic = ann.get("in_contact_object")
                if ic and ic not in _NOT_CONTACT and id_to_ann.get(ic):
                    sample_contact_frame = frame
                    break
            if sample_contact_frame:
                break
        if sample_contact_frame:
            break

    if sample_contact_frame:
        img = sample_contact_frame["image"]
        anns      = sample_contact_frame["annotations"]
        id_to_ann = {a["id"]: a for a in anns}
        print(f"  Video : {img.get('video')}")
        print(f"  Frame : {img.get('name')}")
        print(f"  Annotations in frame: {len(anns)}")
        for ann in anns:
            ic   = ann.get("in_contact_object", "—")
            segs = len(ann.get("segments", []))
            obj_name = ""
            if ic and ic not in _NOT_CONTACT:
                obj_ann = id_to_ann.get(ic)
                obj_name = f" → '{obj_ann.get('name', '?')}'" if obj_ann else " → ?"
            print(f"    id={str(ann['id'])[:8]:<10} name='{ann['name']:<22}' "
                  f"in_contact={str(ic)[:8]:<10} segs={segs}{obj_name}")
    else:
        print("  (No contact frame found — check annotation JSONs)")

    # ── Summary ───────────────────────────────────────────────────────────────
    has_frames = FRAMES_ROOT.exists() and total_jpgs > 0
    has_masks  = MASKS_ROOT.exists() and hand_masks > 0
    has_annos  = (FINAL_ANNO_DIR / "train.json").exists()

    print(f"\n{'='*68}")
    print(f"  Readiness check:")
    print(f"    Contact frames in catalog : {'✓' if contact_frames >= 50 else '⚠'}  "
          f"{contact_frames:,}")
    print(f"    Frames on disk            : {'✓' if has_frames else '✗'}")
    print(f"    Masks on disk             : {'✓' if has_masks else '✗'}")
    print(f"    Final annotation JSONs    : {'✓' if has_annos else '✗'}")

    ready = contact_frames >= 50 and has_frames and has_masks and has_annos
    if ready:
        print(f"\n  ✓ READY — pass to training:")
        print(f"    python train.py --annotation-path {FINAL_ANNO_DIR}/train.json")
    elif contact_frames == 0:
        print(f"\n  ✗ NOT READY — no contact frames in annotations.")
    else:
        missing = []
        if not has_frames: missing.append("frames")
        if not has_masks:  missing.append("masks")
        if not has_annos:  missing.append("annotations")
        print(f"\n  ⚠ PARTIAL — missing: {', '.join(missing)}")
        print(f"    Re-run: python scripts/epic_kitchen/download_epic_kitchen --output-dir {DATA}")
    print("=" * 68)

    return ready


def main() -> None:
    p = argparse.ArgumentParser(
        description="Sanity-check the downloaded VISOR / EPIC Kitchen data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir",    default="./data/epic_kitchen",
                   help="Root output dir from download_epic_kitchen package")
    p.add_argument("--participant", default=None,
                   help="Filter to one participant (e.g. P01). Default: all.")
    p.add_argument("--split",       default="train", choices=["train", "val"])
    args = p.parse_args()

    verify(data_dir=args.data_dir, split=args.split, participant=args.participant)


if __name__ == "__main__":
    main()
