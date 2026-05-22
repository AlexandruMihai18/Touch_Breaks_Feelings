"""Phase 6 — Extract audio tracks for sampled EPIC-Kitchens-100 videos.

ffmpeg streams directly from the EK100/EK55 HTTP endpoint and copies the
embedded AAC track to audio/{video_id}.m4a (AAC in MP4 container) without
downloading the full video to disk.
"""

import shutil
import subprocess
from pathlib import Path

import urllib.request

from .constants import EK100_VIDEO_BASE, EK55_VIDEO_BASE
from .failure_log import FailureLog

_AUDIO_EXT      = "m4a"  # AAC in MP4 container — readable by soundfile/librosa without ffmpeg fallback
_FFMPEG_TIMEOUT = 3600   # seconds per video — EK100 videos run up to ~30 min; ffmpeg ~2× realtime


def audio_path_for(video_id: str, audio_root: Path) -> Path:
    return audio_root / f"{video_id}.{_AUDIO_EXT}"


def _is_ek55(video_id: str) -> bool:
    """Return True for EK55 video IDs (2-digit suffix, e.g. P01_01).

    EK100-only videos use a 3-digit suffix (P01_101+).  Both appear in VISOR
    annotations but live on different HTTP servers.
    """
    suffix = video_id.split("_", 1)[-1]
    return len(suffix) <= 2


def _video_url(video_id: str) -> str:
    pid = video_id[:3]  # e.g. "P01"
    if not _is_ek55(video_id):
        return f"{EK100_VIDEO_BASE}/{pid}/videos/{video_id}.MP4"
    # EK55 videos are split across train/ and test/ on a separate server.
    for split in ("train", "test"):
        url = f"{EK55_VIDEO_BASE}/videos/{split}/{pid}/{video_id}.MP4"
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    return url
        except Exception:
            continue
    # Return the train URL as a best-effort fallback (ffmpeg will fail with a
    # useful error rather than silently returning a wrong result).
    return f"{EK55_VIDEO_BASE}/videos/train/{pid}/{video_id}.MP4"


def _audio_ok(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _extract(video_id: str, audio_root: Path, flog: FailureLog) -> bool:
    dest = audio_path_for(video_id, audio_root)
    if _audio_ok(dest):
        return True

    audio_root.mkdir(parents=True, exist_ok=True)
    url = _video_url(video_id)
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", url, "-vn", "-acodec", "copy", str(dest)],
            capture_output=True,
            timeout=_FFMPEG_TIMEOUT,
        )
        if r.returncode == 0 and _audio_ok(dest):
            return True
        stderr_tail = r.stderr.decode(errors="replace")[-400:]
        flog.record("audio", video_id, dest.name, f"ffmpeg exit {r.returncode}: {stderr_tail}")
        if dest.exists():
            dest.unlink()
        return False
    except subprocess.TimeoutExpired:
        flog.record("audio", video_id, dest.name,
                    f"ffmpeg timed out after {_FFMPEG_TIMEOUT}s")
        if dest.exists():
            dest.unlink()
        return False


def _check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH — install ffmpeg to enable audio download"
        )
    r = subprocess.run(["ffmpeg", "-protocols"], capture_output=True, text=True)
    if "https" not in r.stdout.split():
        raise RuntimeError(
            "ffmpeg on this system was compiled without HTTPS/TLS support.\n"
            "On SLURM, try a different module — e.g.:\n"
            "  module spider FFmpeg          # list available builds\n"
            "  module load FFmpeg/6.0-GCCcore-12.3.0-HTTPS  # if available\n"
            "Or install a TLS-enabled ffmpeg locally:\n"
            "  conda install -c conda-forge ffmpeg\n"
            "  apt-get install ffmpeg        # Ubuntu — includes gnutls"
        )


def download_audio(sampled: list[dict], audio_root: Path, flog: FailureLog) -> None:
    """Extract audio for every unique video_id in *sampled*."""
    _check_ffmpeg()

    video_ids = sorted({rec["video_id"] for rec in sampled})
    ok = skip = fail = 0
    print(f"\n  {len(video_ids)} unique video(s) …")

    for video_id in video_ids:
        dest = audio_path_for(video_id, audio_root)
        if _audio_ok(dest):
            print(f"  {video_id}: already on disk [skip]")
            skip += 1
            continue

        print(f"  {video_id}: streaming audio …", end=" ", flush=True)
        if _extract(video_id, audio_root, flog):
            print("✓")
            ok += 1
        else:
            print("✗")
            fail += 1

    print(f"\n  Done: {ok} extracted, {skip} skipped, {fail} failed")
    if fail:
        print(f"  ⚠ {fail} video(s) failed — see failure log for details")
