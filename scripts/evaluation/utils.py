"""Shared utilities for evaluation scripts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


_ANNOTATION_FIELDS = ["depth_touch", "object_coverage", "x_touch", "y_touch", "object_name"]


def enrich_df(df: pd.DataFrame, annotation_paths: list[Path]) -> pd.DataFrame:
    """Join annotation fields into a predictions DataFrame.

    Matches on (video_dir, image filename, sample type) — the parent directory
    name + basename of frame_path in the CSV against image_path in the
    annotation JSON, with label 1 mapped to type "touch" and label 0 to
    "no-touch". Using the parent directory disambiguates datasets like Kubric
    or Greatest Hits where frame filenames (frame_000001.jpg) repeat across
    videos.

    Fields added / filled: depth_touch, object_coverage, x_touch, y_touch.
    """
    lookup: dict[tuple[str, str, int], dict] = {}
    for path in annotation_paths:
        if not path.exists():
            print(f"[WARN] annotation file not found: {path}")
            continue
        with open(path) as f:
            entries = json.load(f)
        for entry in entries:
            img = entry.get("image_path", "")
            if not img:
                continue
            label = 1 if entry.get("type") == "touch" else 0
            p = Path(img)
            key = (p.parent.name, p.name, label)
            if key not in lookup:
                lookup[key] = entry

    if not lookup:
        print("[WARN] no annotation entries loaded — CSV will not be enriched")
        return df

    df = df.copy()
    for field in _ANNOTATION_FIELDS:
        if field not in df.columns:
            df[field] = None

    matched = unmatched = 0
    for idx, row in df.iterrows():
        p = Path(str(row["frame_path"]))
        key = (p.parent.name, p.name, int(row["label"]))
        entry = lookup.get(key)
        if entry is None:
            unmatched += 1
            continue
        matched += 1
        for field in _ANNOTATION_FIELDS:
            if field in entry:
                df.at[idx, field] = entry[field]

    print(f"enrich_df: matched={matched}  unmatched={unmatched}  total={len(df)}")
    return df
