from __future__ import annotations

import numpy as np

try:
    import cv2

    _has_cv2 = True
except ImportError:
    _has_cv2 = False

DEFAULT_N_FRAMES = 5


def sample_frame_indices(
    total_frames: int, n_frames: int, seed: int | None = None
) -> list[int]:
    """Return n_frames pseudo-uniformly spaced frame indices."""
    rng = np.random.default_rng(seed)

    if total_frames <= n_frames:
        return list(range(total_frames))

    segment = total_frames // n_frames
    indices = []
    for i in range(n_frames):
        start = i * segment
        end = min(start + segment - 1, total_frames - 1)
        indices.append(int(rng.integers(start, end + 1)))

    return sorted(indices)


def sample_items(items: list, n_items: int) -> list:
    """Sample up to n_items entries uniformly across a list."""
    if not items:
        return []

    indices = sample_frame_indices(len(items), min(n_items, len(items)))
    return [items[i] for i in indices]


def sample_video_frames(video_path, n_frames: int = DEFAULT_N_FRAMES):
    """Sample N frames pseudo-uniformly from a video file."""
    if video_path is None:
        return [None] * n_frames

    if not _has_cv2:
        import gradio as gr

        raise gr.Error("opencv-python is required for video frame sampling.")

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return [None] * n_frames

    indices = sample_frame_indices(total, min(n_frames, total))
    if indices and len(indices) < n_frames:
        indices.extend([indices[-1]] * (n_frames - len(indices)))
    elif not indices:
        indices = [0] * n_frames

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret else None)

    cap.release()
    return frames
