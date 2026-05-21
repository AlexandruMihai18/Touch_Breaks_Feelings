import time

import requests

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "VISOR-Downloader/2.0"

# Status codes that are worth retrying (server-side / transient).
# 404 is intentionally excluded — it means the resource doesn't exist.
_RETRYABLE = {429, 500, 502, 503, 504}


def http_get(url: str, stream: bool = False, retries: int = 3, **kwargs):
    for attempt in range(retries):
        try:
            return SESSION.get(url, stream=stream, timeout=30, **kwargs)
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def http_head(url: str, retries: int = 3, **kwargs) -> requests.Response:
    """HEAD with exponential-backoff retry on 5xx / connection errors."""
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            r = SESSION.head(url, timeout=15, **kwargs)
            if r.status_code not in _RETRYABLE:
                return r
            last_exc = None
        except requests.RequestException as exc:
            last_exc = exc
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    if last_exc:
        raise last_exc
    return r  # last response (still a retryable status — caller decides)


def http_range_get(url: str, byte_range: str, retries: int = 3, **kwargs) -> requests.Response:
    """GET with a Range header and exponential-backoff retry."""
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            r = SESSION.get(url, headers={"Range": byte_range}, **kwargs)
            if r.status_code not in _RETRYABLE:
                return r
            last_exc = None
        except requests.RequestException as exc:
            last_exc = exc
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    if last_exc:
        raise last_exc
    return r
