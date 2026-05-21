from pathlib import Path

from .constants import ANNO_BASE, MAX_VIDEO_SUFFIX
from .http import http_get


def fetch_annotations(participants: list[str], split: str, anno_dir: Path) -> list[Path]:
    """
    Download per-video annotation JSONs for each participant from data.bris.ac.uk.

    Video IDs are probed sequentially (P01_01 … P01_130); 404s are silently
    skipped. Files already on disk are reused without re-fetching.

    Returns the list of local JSON paths that are ready to use.
    """
    anno_dir.mkdir(parents=True, exist_ok=True)
    found = []

    for pid in participants:
        print(f"\n  Fetching annotations for {pid} / {split} …")
        for suffix in range(1, MAX_VIDEO_SUFFIX + 1):
            video_id = f"{pid}_{suffix:02d}"
            dest = anno_dir / f"{video_id}.json"

            if dest.exists() and dest.stat().st_size > 100:
                found.append(dest)
                continue

            r = http_get(f"{ANNO_BASE}/{split}/{video_id}.json")
            if r.status_code == 200:
                dest.write_bytes(r.content)
                found.append(dest)
                print(f"    ✓ {video_id}.json")
            elif r.status_code != 404:
                print(f"    ⚠ HTTP {r.status_code} for {video_id}")

    print(f"\n  → {len(found)} annotation JSONs ready")
    return found
