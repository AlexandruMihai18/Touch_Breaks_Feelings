"""Shared utilities for evaluation scripts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


_ANNOTATION_FIELDS = ["depth_touch", "object_coverage", "x_touch", "y_touch"]


def enrich_df(df: pd.DataFrame, annotation_paths: list[Path]) -> pd.DataFrame:
    """Join annotation fields into a predictions DataFrame.

    Matches on (image filename, sample type) — frame_path basename in the CSV
    against image_path basename in the annotation JSON, with label 1 mapped to
    type "touch" and label 0 to "no-touch". Falls back gracefully when fields
    are already present in the CSV.

    Fields added / filled: depth_touch, object_coverage, x_touch, y_touch.
    """
    lookup: dict[tuple[str, int], dict] = {}
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
            key = (Path(img).name, label)
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
        key = (Path(str(row["frame_path"])).name, int(row["label"]))
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
