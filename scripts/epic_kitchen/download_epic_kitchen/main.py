import argparse
import random
import sys
from pathlib import Path

from .constants import ALL_PARTICIPANTS
from .failure_log import FailureLog
from .phase1_annotations import fetch_annotations
from .phase2_catalog import build_catalog, select_frames, stratified_sample
from .phase3_frames import count_on_disk, download_frames
from .phase4_masks import render_all_masks
from .phase5_generate import run_generate_annotations
from .verify_epic_kitchen import verify

_DESCRIPTION = """\
VISOR contact-frame downloader + mask renderer

Phases:
  1. Download annotation JSONs from data.bris.ac.uk
  2. Parse annotations → build contact / no-contact catalog; select frames
  3. Download sampled frames from per-video ZIPs via HTTP range requests
  4. Render VISOR polygon segments → binary hand / object / touch masks
  5. Run generate_annotations.py → annotations/train.json + val.json

Output layout:
  <output_dir>/
  ├── visor/GroundTruth-SparseAnnotations/annotations/{split}/
  ├── frames/{video_id}/{frame_name}.jpg
  ├── masks/{video_id}/{stem}_{hand|object|touch}.png
  ├── annotations/{train|val}.json
  └── failures.json
"""


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=_DESCRIPTION,
    )
    p.add_argument("--output-dir",       default="./data/epic_kitchen")
    p.add_argument("--participants",      nargs="+", default=["P01"],
                   help="Participant IDs (e.g. P01 P02). Ignored with --all-participants.")
    p.add_argument("--all-participants",  action="store_true",
                   help="Download all known VISOR participants")
    p.add_argument("--split",            default="train", choices=["train", "val"])
    p.add_argument("--n-contact",        type=int, default=300)
    p.add_argument("--n-no-contact",     type=int, default=100)
    p.add_argument("--all",              action="store_true",
                   help="Download every matched frame (overrides --n-contact / --n-no-contact)")
    p.add_argument("--match-no-contact", action="store_true",
                   help="Download all contact frames first, then stratified-sample "
                        "no-contact frames to match the number that actually "
                        "succeeded on disk (by video). Overrides all count flags.")
    p.add_argument("--seed",             type=int, default=42)
    p.add_argument("--workers",          type=int, default=4,
                   help="Parallel threads for mask rendering")
    p.add_argument("--val-frac",         type=float, default=0.15)
    p.add_argument("--failure-log",      default=None,
                   help="Path to failure log (default: <output_dir>/failures.json)")
    p.add_argument("--dry-run",          action="store_true",
                   help="Parse + select only; skip download and mask rendering")
    p.add_argument("--skip-annotations", action="store_true",
                   help="Skip Phase 5 (generate_annotations.py)")
    p.add_argument("--skip-verify", action="store_true",
                   help="Skip post-download verification check")
    return p.parse_args()


def main():
    args = parse_args()
    out  = Path(args.output_dir)

    participants     = ALL_PARTICIPANTS if args.all_participants else args.participants
    failure_log_path = Path(args.failure_log) if args.failure_log else out / "failures.json"
    flog             = FailureLog(failure_log_path)

    anno_dir    = out / "visor" / "GroundTruth-SparseAnnotations" / "annotations" / args.split
    frames_root = out / "frames"
    masks_root  = out / "masks"

    if args.match_no_contact:
        mode_str = "all contact  →  match no-contact to successful downloads"
    elif args.all:
        mode_str = "ALL frames (no sampling)"
    else:
        mode_str = f"{args.n_contact} contact + {args.n_no_contact} no-contact"

    print("=" * 68)
    print("  VISOR Downloader + Mask Renderer")
    print(f"  participants : {'ALL' if args.all_participants else participants}")
    print(f"  split        : {args.split}")
    print(f"  mode         : {mode_str}")
    print(f"  output       : {out}")
    print(f"  failure log  : {failure_log_path}")
    print("=" * 68)

    # Phase 1
    print("\n── Phase 1: Download annotation JSONs ──────────────────────────────")
    json_paths = fetch_annotations(participants, args.split, anno_dir)
    if not json_paths:
        print("ERROR: No annotation JSONs. Check network access to data.bris.ac.uk")
        sys.exit(1)

    # Phase 2
    print("\n── Phase 2: Parse + select frames ──────────────────────────────────")
    catalog = build_catalog(json_paths, args.split)

    if args.dry_run:
        select_frames(catalog, args.n_contact, args.n_no_contact,
                      args.seed, take_all=(args.all or args.match_no_contact))
        print("\n  --dry-run: stopping before downloads.")
        return

    if args.match_no_contact:
        # Phase 3a: download every available contact frame
        contact_frames = catalog["contact"]
        print(f"\n  --match-no-contact: {len(contact_frames)} contact frames in catalog")

        print("\n── Phase 3a: Download all contact frames ───────────────────────────")
        download_frames(contact_frames, frames_root, args.split, flog)

        n_ok = count_on_disk(contact_frames, frames_root)
        print(f"\n  {n_ok} / {len(contact_frames)} contact frames on disk — "
              f"sampling {n_ok} no-contact frames to match")

        # Phase 3b: sample no-contact to match confirmed contact count
        rng = random.Random(args.seed)
        no_contact_frames = stratified_sample(
            catalog["no_contact"], n_ok, ["video_id", "participant"], rng
        )

        print("\n── Phase 3b: Download matched no-contact frames ────────────────────")
        download_frames(no_contact_frames, frames_root, args.split, flog)

        sampled = contact_frames + no_contact_frames
    else:
        sampled = select_frames(catalog, args.n_contact, args.n_no_contact,
                                args.seed, take_all=args.all)

        print("\n── Phase 3: Download frames from ZIPs ──────────────────────────────")
        download_frames(sampled, frames_root, args.split, flog)

    # Phase 4
    print("\n── Phase 4: Render masks ───────────────────────────────────────────")
    render_all_masks(sampled, frames_root, masks_root, args.workers, flog)

    # Phase 5
    if not args.skip_annotations:
        print("\n── Phase 5: Generate train/val annotation JSONs ────────────────────")
        run_generate_annotations(out, args.val_frac)

    flog.flush()
    print("\n── Failure summary ─────────────────────────────────────────────────")
    flog.summary()

    if not args.skip_verify:
        print("\n── Verification ────────────────────────────────────────────────────")
        verify(data_dir=out, split=args.split)

    print(f"""
{'='*68}
  Complete.

  {out}/
  ├── frames/{{video_id}}/{{frame_name}}.jpg
  ├── masks/{{video_id}}/
  │   ├── {{stem}}_hand.png
  │   ├── {{stem}}_object.png
  │   └── {{stem}}_touch.png
  ├── annotations/
  │   ├── train.json
  │   └── val.json
  └── failures.json

  Pass to training:
    python train.py --annotation-path {out}/annotations/train.json
{'='*68}""")
