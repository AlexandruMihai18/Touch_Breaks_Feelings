"""
Compare two EPIC-Kitchen data directories and report what needs to be synced.

Usage (on Snellius login node):
    python scripts/epic_kitchen/compare_data_dirs.py \
        --src ~/Touch_Breaks_Feelings/data/epic_kitchen \
        --dst /scratch-shared/dotero/epic_kitchen
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _pct(n, total):
    return f"{100*n/total:.1f}%" if total else "—"


def _count_mask_type(mask_dir: Path, suffix: str) -> int:
    return sum(1 for _ in mask_dir.rglob(f"*{suffix}")) if mask_dir.exists() else 0


def survey(root: Path) -> dict:
    s = {}

    # video IDs present in frames/ and masks/
    frames_dir = root / "frames"
    masks_dir  = root / "masks"
    s["frame_videos"] = set(d.name for d in frames_dir.iterdir() if d.is_dir()) \
                        if frames_dir.exists() else set()
    s["mask_videos"]  = set(d.name for d in masks_dir.iterdir()  if d.is_dir()) \
                        if masks_dir.exists() else set()

    # frame and mask file counts
    s["n_frames"] = sum(1 for _ in frames_dir.rglob("*.jpg")) if frames_dir.exists() else 0
    s["n_masks_raw"]     = _count_mask_type(masks_dir, "_touch.png") + \
                           _count_mask_type(masks_dir, "_hand.png") + \
                           _count_mask_type(masks_dir, "_object.png")
    s["n_depth"]         = _count_mask_type(masks_dir, "_depth.png")
    s["n_refined"]       = _count_mask_type(masks_dir, "_touch_refined.png")

    # audio
    audio_dir = root / "audio"
    s["audio_videos"] = set(f.stem for f in audio_dir.glob("*.m4a")) \
                        if audio_dir.exists() else set()

    # annotations
    anno_dir = root / "annotations"
    for split in ("train", "val"):
        p = anno_dir / f"{split}.json"
        if p.exists():
            pairs = json.loads(p.read_text())
            n_touch   = sum(1 for e in pairs if e.get("type") == "touch")
            n_depth_s = sum(1 for e in pairs if e.get("depth_touch") is not None)
            n_cov     = sum(1 for e in pairs if e.get("object_coverage") is not None)
            s[split] = {
                "pairs":    len(pairs),
                "touch":    n_touch,
                "notouch":  len(pairs) - n_touch,
                "videos":   len({e["video_id"] for e in pairs}),
                "depth_stat":    n_depth_s,
                "coverage_stat": n_cov,
            }
        else:
            s[split] = None

    return s


def print_survey(label: str, root: Path, s: dict) -> None:
    print(f"\n{'═'*60}")
    print(f"  {label}")
    print(f"  {root}")
    print(f"{'═'*60}")

    fv = s["frame_videos"]
    mv = s["mask_videos"]
    av = s["audio_videos"]
    print(f"\n  frames/   : {s['n_frames']:>7,} files  across {len(fv):>3} videos")
    print(f"  masks/    : raw={s['n_masks_raw']:,}  depth={s['n_depth']:,}  refined={s['n_refined']:,}")
    print(f"              across {len(mv):>3} videos")
    print(f"  audio/    : {len(av):>3} .m4a files")

    for split in ("train", "val"):
        d = s[split]
        if d is None:
            print(f"\n  {split}.json  : MISSING")
            continue
        print(f"\n  {split}.json  : {d['pairs']:,} pairs  "
              f"({d['touch']:,} touch  {d['notouch']:,} no-touch  "
              f"{d['videos']} videos)")
        print(f"               depth_stat={d['depth_stat']:,}  "
              f"coverage_stat={d['coverage_stat']:,}")


def print_diff(src_label, src, dst_label, dst, src_root: Path, dst_root: Path) -> None:
    print(f"\n{'═'*60}")
    print(f"  Diff:  {src_label}  →  {dst_label}")
    print(f"{'═'*60}")

    # video-level diffs
    for kind, sk, dk in [
        ("frames", "frame_videos", "frame_videos"),
        ("masks",  "mask_videos",  "mask_videos"),
        ("audio",  "audio_videos", "audio_videos"),
    ]:
        only_src = src[sk] - dst[dk]
        only_dst = dst[dk] - src[sk]
        in_both  = src[sk] & dst[dk]
        print(f"\n  {kind}/")
        print(f"    in both          : {len(in_both)}")
        print(f"    only in {src_label:<10}: {len(only_src)}"
              + (f"  ← needs copy" if only_src else ""))
        if only_src:
            for v in sorted(only_src)[:10]:
                print(f"      {v}")
            if len(only_src) > 10:
                print(f"      … and {len(only_src)-10} more")
        print(f"    only in {dst_label:<10}: {len(only_dst)}"
              + (f"  ← exists in dst only" if only_dst else ""))

    # refined mask diff (file-level estimate via count)
    print(f"\n  refined masks (_depth.png / _touch_refined.png)")
    print(f"    {src_label}: depth={src['n_depth']:,}  refined={src['n_refined']:,}")
    print(f"    {dst_label}: depth={dst['n_depth']:,}  refined={dst['n_refined']:,}")
    if src["n_depth"] > dst["n_depth"] or src["n_refined"] > dst["n_refined"]:
        print(f"    ← {src_label} has more refined masks, include masks/ in sync")

    # annotation diff
    print(f"\n  annotations/")
    for split in ("train", "val"):
        sd = src[split]
        dd = dst[split]
        if sd is None and dd is None:
            print(f"    {split}.json : missing in both")
        elif dd is None:
            print(f"    {split}.json : MISSING in {dst_label}  ← needs copy")
        elif sd is None:
            print(f"    {split}.json : MISSING in {src_label}  (dst has {dd['pairs']:,} pairs)")
        else:
            diff = sd["pairs"] - dd["pairs"]
            if diff == 0:
                print(f"    {split}.json : identical pair count ({sd['pairs']:,})")
            else:
                sign = "+" if diff > 0 else ""
                print(f"    {split}.json : {src_label}={sd['pairs']:,}  {dst_label}={dd['pairs']:,}  "
                      f"(diff {sign}{diff})  ← copy {src_label} version")

    # rsync suggestion
    print(f"\n{'─'*60}")
    print(f"  Suggested rsync (dry-run first, then remove -n):")
    print(f"")
    print(f"    rsync -avn --progress \\")
    print(f"        {src_root}/frames/   {dst_root}/frames/")
    print(f"    rsync -avn --progress \\")
    print(f"        {src_root}/masks/    {dst_root}/masks/")
    print(f"    rsync -avn --progress \\")
    print(f"        {src_root}/audio/    {dst_root}/audio/")
    print(f"    rsync -avn --progress \\")
    print(f"        {src_root}/annotations/  {dst_root}/annotations/")
    print(f"    rsync -avn --progress \\")
    print(f"        {src_root}/visor/    {dst_root}/visor/")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare two EPIC-Kitchen data directories",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--src", required=True, help="Source directory (e.g. ~/Touch_Breaks_Feelings/data/epic_kitchen)")
    ap.add_argument("--dst", required=True, help="Destination directory (e.g. /scratch-shared/dotero/epic_kitchen)")
    ap.add_argument("--src-label", default="local",   help="Label for source")
    ap.add_argument("--dst-label", default="scratch",  help="Label for destination")
    args = ap.parse_args()

    src_root = Path(args.src).expanduser().resolve()
    dst_root = Path(args.dst).expanduser().resolve()

    print(f"Surveying {src_root} …")
    src = survey(src_root)
    print(f"Surveying {dst_root} …")
    dst = survey(dst_root)

    print_survey(args.src_label, src_root, src)
    print_survey(args.dst_label, dst_root, dst)
    print_diff(args.src_label, src, args.dst_label, dst, src_root, dst_root)


if __name__ == "__main__":
    main()