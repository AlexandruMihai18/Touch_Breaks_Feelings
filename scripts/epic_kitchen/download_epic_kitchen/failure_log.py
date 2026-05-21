import datetime
import json
import threading
from collections import defaultdict
from pathlib import Path


class FailureLog:
    """
    Thread-safe persistent failure log.

    Failures are collected in memory during a run and flushed to a JSON file
    at the end, appending to any failures recorded by previous runs.

    Each entry has: type, video_id, frame_name, detail, timestamp.
    """

    def __init__(self, path: Path):
        self._path = path
        self._lock = threading.Lock()
        self._new: list[dict] = []
        self._prior: list[dict] = []
        if path.exists():
            try:
                self._prior = json.loads(path.read_text())
            except Exception:
                self._prior = []

    def record(self, failure_type: str, video_id: str, frame_name: str, detail: str):
        entry = {
            "type":       failure_type,  # "download" | "mask"
            "video_id":   video_id,
            "frame_name": frame_name,
            "detail":     detail,
            "timestamp":  datetime.datetime.now().isoformat(timespec="seconds"),
        }
        with self._lock:
            self._new.append(entry)

    def flush(self):
        """Write this run's failures to disk, appending to prior runs."""
        if not self._new:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        combined = self._prior + self._new
        self._path.write_text(json.dumps(combined, indent=2))
        self._prior = combined
        self._new = []

    def summary(self):
        total = len(self._prior) + len(self._new)
        new   = len(self._new)
        if total == 0:
            print("  No failures recorded.")
            return
        by_type: dict = defaultdict(int)
        for e in self._prior + self._new:
            by_type[e["type"]] += 1
        print(f"  Failures this run : {new}")
        print(f"  Total in log      : {total}  (at {self._path})")
        for t, n in sorted(by_type.items()):
            print(f"    {t:<12}: {n}")
