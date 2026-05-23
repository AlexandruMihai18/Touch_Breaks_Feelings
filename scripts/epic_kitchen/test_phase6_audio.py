"""Localized tests for phase6_audio URL resolution and audio extraction.

Run from the repo root:
    python -m pytest scripts/epic_kitchen/test_phase6_audio.py -v
or directly:
    python scripts/epic_kitchen/test_phase6_audio.py
"""

import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import importlib.util, types

# Load phase6_audio without triggering cv2/heavy deps through __init__
_PKG = "download_epic_kitchen"
_pkg_root = Path(__file__).parent / _PKG

def _stub_module(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m

_stub_module(
    f"{_PKG}.constants",
    EK100_VIDEO_BASE="https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m",
    EK55_VIDEO_BASE ="https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d",
)
_stub_module(f"{_PKG}.failure_log", FailureLog=MagicMock)

# Register a minimal package so relative imports resolve
_pkg = types.ModuleType(_PKG)
_pkg.__path__ = [str(_pkg_root)]
_pkg.__package__ = _PKG
sys.modules[_PKG] = _pkg

_spec = importlib.util.spec_from_file_location(
    f"{_PKG}.phase6_audio", _pkg_root / "phase6_audio.py",
    submodule_search_locations=[],
)
_phase6 = importlib.util.module_from_spec(_spec)
_phase6.__package__ = _PKG
sys.modules[f"{_PKG}.phase6_audio"] = _phase6
_spec.loader.exec_module(_phase6)

_video_url     = _phase6._video_url
_audio_ok      = _phase6._audio_ok
audio_path_for = _phase6.audio_path_for


class TestVideoUrl(unittest.TestCase):
    """_video_url must return a live (HTTP 200) URL for both EK55 and EK100 IDs."""

    def _head_status(self, url: str) -> int:
        r = subprocess.run(
            ["curl", "-sI", "-o", "/dev/null", "-w", "%{http_code}", url],
            capture_output=True, text=True, timeout=15,
        )
        return int(r.stdout.strip())

    def test_ek100_url_resolves(self):
        """EK100 video (3-digit suffix) should return HTTP 200."""
        url = _video_url("P01_101")
        status = self._head_status(url)
        self.assertEqual(status, 200, f"Expected 200, got {status} for {url}")

    def test_ek55_url_resolves(self):
        """EK55 video (2-digit suffix) should return HTTP 200 (train split)."""
        url = _video_url("P01_01")
        status = self._head_status(url)
        self.assertEqual(status, 200, f"Expected 200, got {status} for {url}")

    def test_ek55_test_split_resolves(self):
        """EK55 test-split video (e.g. P01_14) should return HTTP 200."""
        url = _video_url("P01_14")
        status = self._head_status(url)
        self.assertEqual(status, 200, f"Expected 200, got {status} for {url}")

    def test_ek100_url_structure(self):
        """EK100 URL must embed the participant folder and have no split segment."""
        url = _video_url("P02_103")
        self.assertIn("P02/videos/P02_103.MP4", url)

    def test_ek55_url_structure(self):
        """EK55 URL must embed a split segment (train or test)."""
        url = _video_url("P01_01")
        self.assertTrue(
            "/train/" in url or "/test/" in url,
            f"Expected train/ or test/ in EK55 URL, got: {url}",
        )


class TestAudioHelpers(unittest.TestCase):
    def test_audio_ok_missing(self):
        self.assertFalse(_audio_ok(Path("/nonexistent/file.m4a")))

    def test_audio_ok_empty(self, tmp_path=None):
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as f:
            name = f.name
        try:
            self.assertFalse(_audio_ok(Path(name)))  # zero bytes
        finally:
            os.unlink(name)

    def test_audio_path_for(self):
        p = audio_path_for("P01_101", Path("/data/audio"))
        self.assertEqual(p, Path("/data/audio/P01_101.m4a"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
