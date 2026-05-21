import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path

from .constants import FRAME_BASE
from .failure_log import FailureLog
from .http import http_head, http_range_get

_ZIP_TIMEOUT = 360  # seconds — skip a video ZIP that takes longer than this


def _jpeg_ok(path: Path) -> bool:
    """Return True only if path exists, is non-empty, and ends with the JPEG EOI marker."""
    try:
        if not path.exists() or path.stat().st_size < 2:
            return False
        with open(path, "rb") as f:
            f.seek(-2, 2)
            return f.read(2) == b"\xff\xd9"
    except OSError:
        return False


class RangeHTTPFile:
    """
    Seekable file-like object backed by HTTP range requests.

    Pre-fetches the last 64 KB on construction, which covers the ZIP
    end-of-central-directory record and the entire central directory for
    typical VISOR ZIPs (~35 KB). Subsequent reads use a 1 MB read-ahead
    buffer to minimise round trips when zipfile scans local file headers
    and compressed frame data sequentially.
    """
    _READ_AHEAD = 1 << 20  # 1 MB

    def __init__(self, url: str, size: int, tail_size: int = 65536):
        self.url  = url
        self.size = size
        self.pos  = 0
        tail_start = max(0, size - tail_size)
        r = http_range_get(url, f"bytes={tail_start}-{size-1}", timeout=30)
        r.raise_for_status()
        self._tail_start = tail_start
        self._tail = r.content
        self._buf_start = -1
        self._buf: bytes = b""

    def seek(self, pos, whence=0):
        if   whence == 0: self.pos = pos
        elif whence == 1: self.pos += pos
        elif whence == 2: self.pos = self.size + pos
        return self.pos

    def tell(self): return self.pos

    def read(self, n=-1):
        if n < 0: n = self.size - self.pos
        if n <= 0 or self.pos >= self.size: return b""
        n = min(n, self.size - self.pos)

        # Entirely within the pre-cached tail
        if self.pos >= self._tail_start:
            i = self.pos - self._tail_start
            chunk = self._tail[i: i + n]
            self.pos += len(chunk)
            return chunk

        # Read spans the tail boundary — combine pre-tail fetch with tail cache
        if self.pos + n > self._tail_start:
            pre_n = self._tail_start - self.pos
            pre_chunk = self._fetch_pre_tail(pre_n)
            tail_chunk = self._tail[: n - len(pre_chunk)]
            chunk = pre_chunk + tail_chunk
            self.pos += len(chunk)
            return chunk

        # Entirely in the pre-tail region
        chunk = self._fetch_pre_tail(n)
        self.pos += len(chunk)
        return chunk

    def _fetch_pre_tail(self, n: int) -> bytes:
        """Return n bytes from self.pos without advancing pos (caller must advance)."""
        if (self._buf_start >= 0
                and self.pos >= self._buf_start
                and self.pos + n <= self._buf_start + len(self._buf)):
            i = self.pos - self._buf_start
            return self._buf[i: i + n]

        end = min(self.pos + self._READ_AHEAD - 1, self._tail_start - 1)
        r = http_range_get(self.url, f"bytes={self.pos}-{end}", timeout=60)
        r.raise_for_status()
        self._buf_start = self.pos
        self._buf = r.content
        return self._buf[:n]

    def readable(self): return True
    def seekable(self): return True
    def writable(self): return False


def _extract_from_zip(zip_url: str, needed: set[str], dest_dir: Path,
                      video_id: str, flog: FailureLog) -> tuple[int, int]:
    """
    Selectively extract frames from a remote ZIP via HTTP range requests.
    Returns (n_ok, n_fail). All failures are recorded in flog.
    """
    to_fetch = {n for n in needed if not _jpeg_ok(dest_dir / n)}
    if not to_fetch:
        return 0, 0

    try:
        r = http_head(zip_url)
        if r.status_code != 200:
            detail = f"ZIP HEAD returned {r.status_code}: {zip_url}"
            print(f"    ⚠ {detail}")
            for name in to_fetch:
                flog.record("download", video_id, name, detail)
            return 0, len(to_fetch)
        size = int(r.headers.get("Content-Length", 0))
        if not size:
            detail = f"ZIP has no Content-Length: {zip_url}"
            print(f"    ⚠ {detail}")
            for name in to_fetch:
                flog.record("download", video_id, name, detail)
            return 0, len(to_fetch)
    except Exception as e:
        detail = f"ZIP HEAD failed: {e}"
        print(f"    ⚠ {detail}")
        for name in to_fetch:
            flog.record("download", video_id, name, detail)
        return 0, len(to_fetch)

    dest_dir.mkdir(parents=True, exist_ok=True)
    ok = fail = 0
    try:
        f = RangeHTTPFile(zip_url, size)
        with zipfile.ZipFile(f) as zf:
            zip_names = set(zf.namelist())
            for frame_name in to_fetch:
                if frame_name not in zip_names:
                    detail = f"frame not found in ZIP {zip_url}"
                    print(f"    ⚠ Not in ZIP: {frame_name}")
                    flog.record("download", video_id, frame_name, detail)
                    fail += 1
                    continue
                try:
                    data = zf.read(frame_name)
                    (dest_dir / frame_name).write_bytes(data)
                    ok += 1
                except Exception as e:
                    detail = f"extract error: {e}"
                    print(f"    ⚠ Extract failed ({frame_name}): {e}")
                    flog.record("download", video_id, frame_name, detail)
                    fail += 1
    except Exception as e:
        detail = f"ZIP read error: {e}"
        print(f"    ⚠ {detail}")
        for name in list(to_fetch)[ok:]:
            flog.record("download", video_id, name, detail)
        fail += len(to_fetch) - ok

    return ok, fail


def download_frames(sampled: list, frames_root: Path, split: str, flog: FailureLog):
    """Download all frames in *sampled*, one ZIP per video."""
    by_video: dict[str, list] = defaultdict(list)
    for rec in sampled:
        by_video[rec["video_id"]].append(rec)

    total_ok = total_fail = total_skip = 0
    print(f"\n  {len(sampled)} frames across {len(by_video)} videos …")

    for vid, recs in sorted(by_video.items()):
        pid    = recs[0]["participant"]
        needed = {rec["frame_name"] for rec in recs}
        already = sum(1 for n in needed if _jpeg_ok(frames_root / vid / n))
        total_skip += already

        to_fetch = {n for n in needed if not _jpeg_ok(frames_root / vid / n)}
        if not to_fetch:
            print(f"  {vid}: {already} frames already on disk [skip]")
            continue

        zip_url = f"{FRAME_BASE}/{split}/{pid}/{vid}.zip"
        print(f"  {vid}: extracting {len(to_fetch)} frame(s) from ZIP …", end=" ", flush=True)
        with ThreadPoolExecutor(max_workers=1) as _ex:
            _fut = _ex.submit(_extract_from_zip, zip_url, to_fetch,
                              frames_root / vid, vid, flog)
            try:
                ok, fail = _fut.result(timeout=_ZIP_TIMEOUT)
            except FuturesTimeoutError:
                detail = (f"ZIP extraction timed out after "
                          f"{_ZIP_TIMEOUT // 60} min: {zip_url}")
                print(f"\n    ⚠ {detail}")
                for name in to_fetch:
                    flog.record("download", vid, name, detail)
                ok, fail = 0, len(to_fetch)
        print(f"✓{ok}  ✗{fail}")
        total_ok   += ok
        total_fail += fail

    print(f"\n  Done: {total_ok} downloaded, {total_skip} skipped, {total_fail} failed")
    if total_fail:
        print(f"  ⚠ {total_fail} frames failed — see failure log for details")


def count_on_disk(records: list, frames_root: Path) -> int:
    """Return how many unique frame files in *records* are present on disk.

    Deduplicates by (video_id, frame_name) so that multiple contact-pair
    records for the same frame are counted only once.
    """
    unique = {(rec["video_id"], rec["frame_name"]) for rec in records}
    return sum(1 for vid, name in unique if _jpeg_ok(frames_root / vid / name))
