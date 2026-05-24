"""Plot touch-detection errors by temporal offset from a touch frame.

For Greatest Hits and EPIC Kitchen, offset 0 comes from touch rows in the main
annotation JSON.  Nonzero offsets come from <split>_context_frames.json when it
exists; otherwise they are inferred from no-touch rows in the main annotation
JSON by sorting each video by frame_idx and measuring row distance to the
nearest touch row.  For Kubric, offset 0 comes from touch rows and nonzero
offsets come from kubric.hard_negative_offsets on no-touch rows.

By default, the plotted metric is class-specific:
  * offset 0: false negative rate on touch frames
  * nonzero offsets: false positive rate on no-touch context frames

Use --metric f1 to plot binary F1 per offset bin instead.

Usage
-----
    python scripts/evaluation/temporal_offsets/analyze_offset_performance.py \\
        --csv results/predictions.csv \\
        --datasets gh ek kubric \\
        --data-root /gpfs/scratch1/shared/dotero \\
        --split val \\
        --metric error \\
        --output-dir results/evaluation/temporal_offsets
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plot_style


DATASET_SUBDIRS = {
    "gh": Path("greatest_hits/annotations"),
    "ek": Path("epic_kitchen/annotations"),
    "kubric": Path("kubric/annotations"),
}

DATASET_LABELS = {
    "gh": "Greatest Hits",
    "ek": "EPIC Kitchen",
    "kubric": "Kubric",
}

METRICS = ["error", "accuracy", "precision", "recall", "f1"]

_FRAME_NUM_RE = re.compile(r"(?:frame_)?(\d+)(?:\.[^.]+)?$")


@dataclass(frozen=True)
class OffsetRow:
    dataset: str
    video_id: str
    frame_name: str
    label: int
    offset_steps: int


def _read_json(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[WARN] annotation file not found: {path}")
        return []
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        print(f"[WARN] expected list in annotation file: {path}")
        return []
    return data


def _frame_name(path_value: object) -> str:
    return Path(str(path_value)).name


def _video_id(entry: dict) -> str:
    value = entry.get("video_id")
    if value:
        return str(value)
    image_path = entry.get("image_path")
    return Path(str(image_path)).parent.name if image_path else ""


def _label_from_entry(entry: dict) -> int:
    return 1 if entry.get("type") == "touch" else 0


def _frame_idx(entry: dict) -> int | None:
    for key in ("frame_idx", "frame_id", "frame"):
        value = entry.get(key)
        if value is not None and value != "":
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    stem = Path(str(entry.get("image_path", ""))).stem
    match = _FRAME_NUM_RE.search(stem)
    return int(match.group(1)) if match else None


def _offset_row(dataset: str, entry: dict, label: int, offset_steps: int) -> OffsetRow:
    return OffsetRow(
        dataset=dataset,
        video_id=_video_id(entry),
        frame_name=_frame_name(entry.get("image_path", "")),
        label=label,
        offset_steps=offset_steps,
    )


def _touch_offsets_from_main(dataset: str, entries: list[dict]) -> list[OffsetRow]:
    rows: list[OffsetRow] = []
    for entry in entries:
        if entry.get("type") != "touch":
            continue
        rows.append(_offset_row(dataset, entry, label=1, offset_steps=0))
    return rows


def _infer_offsets_from_main(dataset: str, entries: list[dict], max_offset: int) -> list[OffsetRow]:
    rows = _touch_offsets_from_main(dataset, entries)
    by_video: dict[str, list[tuple[int, dict]]] = {}
    for entry in entries:
        idx = _frame_idx(entry)
        if idx is None:
            continue
        by_video.setdefault(_video_id(entry), []).append((idx, entry))

    inferred = 0
    for video_entries in by_video.values():
        ordered = [entry for _, entry in sorted(video_entries, key=lambda item: item[0])]
        touch_positions = [i for i, entry in enumerate(ordered) if entry.get("type") == "touch"]
        if not touch_positions:
            continue

        for i, entry in enumerate(ordered):
            if entry.get("type") == "touch":
                continue
            nearest = min(touch_positions, key=lambda pos: abs(i - pos))
            offset = i - nearest
            if offset == 0 or abs(offset) > max_offset:
                continue
            rows.append(_offset_row(dataset, entry, label=0, offset_steps=offset))
            inferred += 1

    print(f"{dataset}: inferred {inferred} nonzero offsets from {len(entries)} main annotation rows")
    return rows


def _load_standard_offsets(dataset: str, anno_dir: Path, split: str, max_offset: int) -> list[OffsetRow]:
    entries = _read_json(anno_dir / f"{split}.json")
    rows = _touch_offsets_from_main(dataset, entries)

    context_rows = []
    for entry in _read_json(anno_dir / f"{split}_context_frames.json"):
        if entry.get("type") == "touch":
            continue
        offset = entry.get("offset_steps")
        if offset is None:
            continue
        context_rows.append(_offset_row(dataset, entry, label=0, offset_steps=int(offset)))

    if context_rows:
        rows.extend(context_rows)
        return rows

    print(f"{dataset}: no {split}_context_frames.json offsets found; falling back to {split}.json frame_idx order")
    return _infer_offsets_from_main(dataset, entries, max_offset)


def _load_kubric_offsets(dataset: str, anno_dir: Path, split: str) -> list[OffsetRow]:
    rows: list[OffsetRow] = []
    for entry in _read_json(anno_dir / f"{split}.json"):
        label = _label_from_entry(entry)
        if label == 1:
            rows.append(
                OffsetRow(
                    dataset=dataset,
                    video_id=_video_id(entry),
                    frame_name=_frame_name(entry.get("image_path", "")),
                    label=1,
                    offset_steps=0,
                )
            )
            continue

        offsets = (entry.get("kubric") or {}).get("hard_negative_offsets") or []
        for item in offsets:
            if not isinstance(item, dict) or item.get("offset") is None:
                continue
            rows.append(
                OffsetRow(
                    dataset=dataset,
                    video_id=_video_id(entry),
                    frame_name=_frame_name(entry.get("image_path", "")),
                    label=0,
                    offset_steps=int(item["offset"]),
                )
            )
    return rows


def _load_offsets(dataset: str, anno_dir: Path, split: str, max_offset: int) -> list[OffsetRow]:
    if dataset == "kubric":
        return _load_kubric_offsets(dataset, anno_dir, split)
    return _load_standard_offsets(dataset, anno_dir, split, max_offset)


def _clean_int(value: object) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _load_predictions(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _prediction_lookup(rows: list[dict[str, str]]) -> tuple[dict[tuple[str, str, int], int], dict[tuple[str, int], int]]:
    primary: dict[tuple[str, str, int], int] = {}
    fallback: dict[tuple[str, int], int] = {}
    duplicate_primary = duplicate_fallback = 0

    for row in rows:
        label = _clean_int(row.get("label"))
        pred = _clean_int(row.get("prediction"))
        frame_path = row.get("frame_path")
        if label is None or pred is None or frame_path is None:
            continue

        frame = _frame_name(frame_path)
        video = str(row.get("video_id") or "")
        if video:
            key = (video, frame, label)
            if key in primary:
                duplicate_primary += 1
            else:
                primary[key] = pred

        fb_key = (frame, label)
        if fb_key in fallback:
            duplicate_fallback += 1
        else:
            fallback[fb_key] = pred

    if duplicate_primary:
        print(f"[WARN] duplicate primary prediction keys ignored: {duplicate_primary}")
    if duplicate_fallback:
        print(f"[WARN] duplicate fallback prediction keys ignored: {duplicate_fallback}")
    return primary, fallback


def _match_prediction(
    row: OffsetRow,
    primary: dict[tuple[str, str, int], int],
    fallback: dict[tuple[str, int], int],
) -> int | None:
    if row.video_id:
        pred = primary.get((row.video_id, row.frame_name, row.label))
        if pred is not None:
            return pred
    return fallback.get((row.frame_name, row.label))


def _analysis_df(
    rows: Iterable[OffsetRow],
    primary: dict[tuple[str, str, int], int],
    fallback: dict[tuple[str, int], int],
) -> list[dict]:
    records = []
    matched = unmatched = 0
    for row in rows:
        pred = _match_prediction(row, primary, fallback)
        if pred is None:
            unmatched += 1
            continue
        matched += 1
        records.append(
            {
                "dataset": row.dataset,
                "offset_steps": row.offset_steps,
                "offset_abs": abs(row.offset_steps),
                "label": row.label,
                "prediction": pred,
                "error": int(pred != row.label),
            }
        )
    print(f"{next(iter(records), {}).get('dataset', 'dataset')}: matched={matched}  unmatched={unmatched}")
    return records


def _signed_stats(records: list[dict], dataset: str, max_offset: int) -> list[dict]:
    rows = []
    for offset in range(-max_offset, max_offset + 1):
        group = [r for r in records if r["offset_steps"] == offset]
        rows.append(_metric_row(dataset, offset, abs(offset), group))
    return rows


def _absolute_stats(records: list[dict], dataset: str, max_offset: int) -> list[dict]:
    rows = []
    for offset_abs in range(0, max_offset + 1):
        group = [r for r in records if r["offset_abs"] == offset_abs]
        rows.append(_metric_row(dataset, offset_abs, offset_abs, group))
    return rows


def _binary_metrics(group: list[dict]) -> dict[str, float]:
    n = len(group)
    if n == 0:
        return {metric: math.nan for metric in METRICS}

    tp = sum(r["label"] == 1 and r["prediction"] == 1 for r in group)
    tn = sum(r["label"] == 0 and r["prediction"] == 0 for r in group)
    fp = sum(r["label"] == 0 and r["prediction"] == 1 for r in group)
    fn = sum(r["label"] == 1 and r["prediction"] == 0 for r in group)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "error": (fp + fn) / n,
        "accuracy": (tp + tn) / n,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _metric_row(dataset: str, offset_steps: int, offset_abs: int, group: list[dict]) -> dict:
    n = len(group)
    errors = sum(int(r["error"]) for r in group) if n else 0
    return {
        "dataset": dataset,
        "offset_steps": offset_steps,
        "offset_abs": offset_abs,
        "n": n,
        "errors": errors,
        "error_rate": errors / n if n else math.nan,
        "metric_type": "fn_rate" if offset_abs == 0 else "fp_rate",
        **_binary_metrics(group),
    }


def _plot_bars(stats: list[dict], title: str, xlabel: str, metric: str, output: Path) -> None:
    plot_style.apply()
    x = list(range(len(stats)))
    values = [float(row[metric]) for row in stats]
    if metric == "error":
        colors = [plot_style.C_FN if row["metric_type"] == "fn_rate" else plot_style.C_FP for row in stats]
    else:
        colors = plot_style.recall_colors([0 if math.isnan(v) else v for v in values])
    labels = [f"{int(row['offset_steps']):+d}" if xlabel == "Signed offset steps" else str(int(row["offset_steps"])) for row in stats]

    fig, ax = plt.subplots(figsize=(max(5.0, len(stats) * 0.55), 3.8))
    heights = [0 if math.isnan(v) else v for v in values]
    ax.bar(x, heights, color=colors, edgecolor="white", linewidth=0.6, width=0.62, zorder=3)

    for i, row in enumerate(stats):
        if row["n"] == 0 or math.isnan(float(row[metric])):
            ax.text(i, 0.025, "n=0", ha="center", va="bottom", fontsize=8, color=plot_style.GRAY)
            continue
        ax.text(
            i,
            min(float(row[metric]) + 0.03, 1.07),
            f"{float(row[metric]):.2f}\nn={int(row['n'])}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=plot_style.DARK,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Error rate" if metric == "error" else metric.capitalize())
    ax.set_ylim(0, 1.15)
    ax.set_title(title)
    ax.grid(axis="y", zorder=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output)
    plt.close(fig)
    print(f"Saved -> {output}")


def _write_stats_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "dataset",
        "offset_steps",
        "offset_abs",
        "n",
        "errors",
        "error_rate",
        "metric_type",
        "error",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_dataset_outputs(records: list[dict], dataset: str, output_dir: Path, max_offset: int, metric: str) -> None:
    label = DATASET_LABELS.get(dataset, dataset)
    signed = _signed_stats(records, dataset, max_offset)
    absolute = _absolute_stats(records, dataset, max_offset)
    suffix = "errors" if metric == "error" else metric

    stats_path = output_dir / f"{dataset}_offset_stats.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_stats_csv(stats_path, signed)
    print(f"Saved -> {stats_path}")

    _plot_bars(
        signed,
        f"{label}: {_metric_title(metric)} by signed temporal offset",
        "Signed offset steps",
        metric,
        output_dir / f"{dataset}_signed_offset_{suffix}.png",
    )
    _plot_bars(
        absolute,
        f"{label}: {_metric_title(metric)} by absolute temporal offset",
        "Absolute offset steps",
        metric,
        output_dir / f"{dataset}_absolute_offset_{suffix}.png",
    )


def _metric_title(metric: str) -> str:
    return "error" if metric == "error" else metric.upper() if metric == "f1" else metric


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, required=True, help="Predictions CSV with label, prediction, frame_path")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASET_SUBDIRS), default=list(DATASET_SUBDIRS))
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[3] / "data")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[3] / "results" / "evaluation" / "temporal_offsets")
    parser.add_argument("--max-offset", type=int, default=4)
    parser.add_argument(
        "--metric",
        choices=METRICS,
        default="error",
        help="Metric to plot per offset bin. Stats CSV always includes all metrics.",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(args.csv)

    pred_rows = _load_predictions(args.csv)
    required = {"label", "prediction", "frame_path"}
    columns = set(pred_rows[0]) if pred_rows else set()
    if missing := required - columns:
        raise ValueError(f"CSV missing columns: {missing}")

    primary, fallback = _prediction_lookup(pred_rows)

    for dataset in args.datasets:
        anno_dir = args.data_root / DATASET_SUBDIRS[dataset]
        rows = _load_offsets(dataset, anno_dir, args.split, args.max_offset)
        if not rows:
            print(f"[SKIP] {dataset}: no offset annotation rows found in {anno_dir}")
            continue

        records = _analysis_df(rows, primary, fallback)
        if not records:
            print(f"[SKIP] {dataset}: no offset rows matched predictions")
            continue

        records = [r for r in records if r["offset_abs"] <= args.max_offset]
        if not records:
            print(f"[SKIP] {dataset}: no matched rows within max offset {args.max_offset}")
            continue

        _write_dataset_outputs(records, dataset, args.output_dir, args.max_offset, args.metric)


if __name__ == "__main__":
    main()
