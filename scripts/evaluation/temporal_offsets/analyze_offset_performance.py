"""Plot touch-detection errors by temporal offset from a touch frame.

For Greatest Hits and EPIC Kitchen, offset 0 comes from touch rows in the main
annotation JSON.  Nonzero offsets come from <split>_context_frames.json when it
exists; otherwise they are inferred from no-touch rows in the main annotation
JSON by sorting each video by frame_idx and measuring row distance to the
nearest touch row.  For Kubric, offset 0 comes from touch rows and nonzero
offsets come from kubric.hard_negative_offsets when present; otherwise they
fall back to the same frame-order inference.

By default, the plotted metric is class-specific:
  * offset 0: false negative rate on touch frames
  * nonzero offsets: false positive rate on no-touch context frames

Use --metric f1 to plot binary F1 per offset bin instead.
Use --distance-bins or --balanced-bins to additionally plot binned distance
groups.  By default, bins are symmetric: negative and positive ranges stay
separate, with offset 0 included as its own bin.

Usage
-----
    python scripts/evaluation/temporal_offsets/analyze_offset_performance.py \\
        --csv results/model_a.csv results/model_b.csv \\
        --run-names model_a model_b \\
        --datasets gh ek kubric \\
        --data-root /gpfs/scratch1/shared/dotero \\
        --split val \\
        --metric error \\
        --distance-bins 1-4,5-10,11-20 \\
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
from matplotlib.lines import Line2D

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
DEFAULT_DISTANCE_BINS = "1-4,5-10,11-20"

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


def _infer_offsets_from_main(dataset: str, entries: list[dict], max_offset: int | None) -> list[OffsetRow]:
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
            if offset == 0 or (max_offset is not None and abs(offset) > max_offset):
                continue
            rows.append(_offset_row(dataset, entry, label=0, offset_steps=offset))
            inferred += 1

    print(f"{dataset}: inferred {inferred} nonzero offsets from {len(entries)} main annotation rows")
    return rows


def _load_standard_offsets(dataset: str, anno_dir: Path, split: str, max_offset: int | None) -> list[OffsetRow]:
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


def _load_kubric_offsets(dataset: str, anno_dir: Path, split: str, max_offset: int | None) -> list[OffsetRow]:
    entries = _read_json(anno_dir / f"{split}.json")
    rows = _touch_offsets_from_main(dataset, entries)
    explicit_nonzero = 0

    for entry in entries:
        label = _label_from_entry(entry)
        if label == 1:
            continue

        offsets = (entry.get("kubric") or {}).get("hard_negative_offsets") or []
        for item in offsets:
            if not isinstance(item, dict) or item.get("offset") is None:
                continue
            offset = int(item["offset"])
            if offset == 0 or (max_offset is not None and abs(offset) > max_offset):
                continue
            rows.append(_offset_row(dataset, entry, label=0, offset_steps=offset))
            explicit_nonzero += 1

    inferred_rows = _infer_offsets_from_main(dataset, entries, max_offset)
    inferred_nonzero = [row for row in inferred_rows if row.offset_steps != 0]
    if not explicit_nonzero:
        print(f"{dataset}: no kubric.hard_negative_offsets found; using frame_idx-order fallback")
        return inferred_rows

    explicit_keys = {(row.video_id, row.frame_name, row.label, row.offset_steps) for row in rows}
    added = 0
    for row in inferred_nonzero:
        key = (row.video_id, row.frame_name, row.label, row.offset_steps)
        if key in explicit_keys:
            continue
        rows.append(row)
        added += 1
    if added:
        print(f"{dataset}: added {added} frame-order fallback offsets beyond explicit hard negatives")
    return rows


def _load_offsets(dataset: str, anno_dir: Path, split: str, max_offset: int | None) -> list[OffsetRow]:
    if dataset == "kubric":
        return _load_kubric_offsets(dataset, anno_dir, split, max_offset)
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
    run_name: str,
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
                "run": run_name,
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


def _signed_stats(records: list[dict], dataset: str, run_name: str, max_offset: int) -> list[dict]:
    rows = []
    for offset in range(-max_offset, max_offset + 1):
        group = [r for r in records if r["offset_steps"] == offset]
        rows.append(_metric_row(dataset, run_name, offset, abs(offset), group))
    return rows


def _absolute_stats(records: list[dict], dataset: str, run_name: str, max_offset: int) -> list[dict]:
    rows = []
    for offset_abs in range(0, max_offset + 1):
        group = [r for r in records if r["offset_abs"] == offset_abs]
        rows.append(_metric_row(dataset, run_name, offset_abs, offset_abs, group))
    return rows


def _binned_stats(records: list[dict], dataset: str, run_name: str, bins: list[dict]) -> list[dict]:
    rows = []
    for bin_def in bins:
        if bin_def["side"] == "zero":
            group = [r for r in records if r["offset_steps"] == 0]
        elif bin_def["side"] == "neg":
            group = [r for r in records if -bin_def["hi"] <= r["offset_steps"] <= -bin_def["lo"]]
        elif bin_def["side"] == "pos":
            group = [r for r in records if bin_def["lo"] <= r["offset_steps"] <= bin_def["hi"]]
        else:
            group = [r for r in records if bin_def["lo"] <= r["offset_abs"] <= bin_def["hi"]]

        row = _metric_row(dataset, run_name, bin_def["offset_steps"], bin_def["lo"], group)
        row["bin_label"] = bin_def["label"]
        row["bin_lo"] = bin_def["lo"]
        row["bin_hi"] = bin_def["hi"]
        row["bin_side"] = bin_def["side"]
        rows.append(row)
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


def _metric_row(dataset: str, run_name: str, offset_steps: int, offset_abs: int, group: list[dict]) -> dict:
    n = len(group)
    errors = sum(int(r["error"]) for r in group) if n else 0
    return {
        "run": run_name,
        "dataset": dataset,
        "offset_steps": offset_steps,
        "offset_abs": offset_abs,
        "n": n,
        "errors": errors,
        "error_rate": errors / n if n else math.nan,
        "metric_type": "fn_rate" if offset_abs == 0 else "fp_rate",
        **_binary_metrics(group),
    }


def _plot_lines(stats_by_run: dict[str, list[dict]], title: str, xlabel: str, metric: str, output: Path) -> None:
    plot_style.apply()
    first_stats = next(iter(stats_by_run.values()))
    if first_stats and "bin_label" in first_stats[0]:
        labels = [str(row["bin_label"]) for row in first_stats]
    else:
        labels = [f"{int(row['offset_steps']):+d}" if xlabel == "Signed offset steps" else str(int(row["offset_steps"])) for row in first_stats]
    x = list(range(len(first_stats)))

    fig, ax = plt.subplots(figsize=(max(5.6, len(first_stats) * 0.7), 4.1))
    colors = plt.cm.tab10.colors
    for i, (run_name, stats) in enumerate(stats_by_run.items()):
        values = [math.nan if row["n"] == 0 else float(row[metric]) for row in stats]
        y = [math.nan if math.isnan(v) else v for v in values]
        color = colors[i % len(colors)]
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=1.9,
            markersize=4.5,
            color=color,
            label=run_name,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(_ylabel(metric))
    ax.set_ylim(0, 1.15)
    ax.set_title(title)
    ax.grid(axis="y", zorder=0)

    handles, labels_legend = ax.get_legend_handles_labels()
    if metric != "error":
        handles.append(Line2D([], [], linestyle="none", label=f"metric: {_metric_title(metric)} per offset bin"))
    labels_legend = [h.get_label() for h in handles]
    ax.legend(
        handles,
        labels_legend,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=min(3, len(handles)),
        borderaxespad=0.0,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output)
    plt.close(fig)
    print(f"Saved -> {output}")


def _write_stats_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "run",
        "dataset",
        "bin_label",
        "bin_lo",
        "bin_hi",
        "bin_side",
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


def _write_dataset_outputs(records_by_run: dict[str, list[dict]], dataset: str, output_dir: Path, max_offset: int, metric: str) -> None:
    label = DATASET_LABELS.get(dataset, dataset)
    signed_by_run, absolute_by_run = _standard_stats_by_run(records_by_run, dataset, max_offset)
    signed = [row for rows in signed_by_run.values() for row in rows]
    suffix = "errors" if metric == "error" else metric

    stats_path = output_dir / f"{dataset}_offset_stats.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_stats_csv(stats_path, signed)
    print(f"Saved -> {stats_path}")

    _plot_lines(
        signed_by_run,
        f"{label}: {_metric_title(metric)} by signed temporal offset",
        "Signed offset steps",
        metric,
        output_dir / f"{dataset}_signed_offset_{suffix}.png",
    )
    _plot_lines(
        absolute_by_run,
        f"{label}: {_metric_title(metric)} by absolute temporal offset",
        "Absolute offset steps",
        metric,
        output_dir / f"{dataset}_absolute_offset_{suffix}.png",
    )


def _standard_stats_by_run(
    records_by_run: dict[str, list[dict]],
    dataset: str,
    max_offset: int,
) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    signed_by_run = {
        run_name: _signed_stats(records, dataset, run_name, max_offset)
        for run_name, records in records_by_run.items()
    }
    absolute_by_run = {
        run_name: _absolute_stats(records, dataset, run_name, max_offset)
        for run_name, records in records_by_run.items()
    }
    return signed_by_run, absolute_by_run


def _binned_stats_by_run(
    records_by_run: dict[str, list[dict]],
    dataset: str,
    bins: list[dict],
) -> dict[str, list[dict]]:
    return {
        run_name: _binned_stats(records, dataset, run_name, bins)
        for run_name, records in records_by_run.items()
    }


def _write_binned_outputs(
    records_by_run: dict[str, list[dict]],
    dataset: str,
    output_dir: Path,
    metric: str,
    bins: list[dict],
    suffix_label: str,
) -> None:
    label = DATASET_LABELS.get(dataset, dataset)
    binned_by_run = _binned_stats_by_run(records_by_run, dataset, bins)
    rows = [row for run_rows in binned_by_run.values() for row in run_rows]
    suffix = "errors" if metric == "error" else metric
    stats_path = output_dir / f"{dataset}_{suffix_label}_offset_stats.csv"
    _write_stats_csv(stats_path, rows)
    print(f"Saved -> {stats_path}")
    _plot_lines(
        binned_by_run,
        f"{label}: {_metric_title(metric)} by binned touch distance",
        "Absolute distance from touch",
        metric,
        output_dir / f"{dataset}_{suffix_label}_offset_{suffix}.png",
    )


def _plot_combined_grid(
    stats_by_dataset: dict[str, dict[str, list[dict]]],
    title: str,
    xlabel: str,
    metric: str,
    output: Path,
) -> None:
    if not stats_by_dataset:
        return

    plot_style.apply()
    n = len(stats_by_dataset)
    fig_h = max(3.1, 2.6 * n)
    fig, axes = plt.subplots(n, 1, figsize=(10.5, fig_h), squeeze=False)
    colors = plt.cm.tab10.colors
    legend_handles: list[Line2D] = []

    for ax, (dataset, stats_by_run) in zip(axes[:, 0], stats_by_dataset.items()):
        first_stats = next(iter(stats_by_run.values()))
        if first_stats and "bin_label" in first_stats[0]:
            labels = [str(row["bin_label"]) for row in first_stats]
        else:
            labels = [
                f"{int(row['offset_steps']):+d}" if xlabel == "Signed offset steps" else str(int(row["offset_steps"]))
                for row in first_stats
            ]
        x = list(range(len(first_stats)))

        for i, (run_name, stats) in enumerate(stats_by_run.items()):
            color = colors[i % len(colors)]
            values = [math.nan if row["n"] == 0 else float(row[metric]) for row in stats]
            ax.plot(
                x,
                values,
                marker="o",
                linewidth=1.7,
                markersize=4.0,
                color=color,
                label=run_name,
                zorder=3,
            )
            if len(legend_handles) < len(stats_by_run):
                legend_handles.append(Line2D([], [], color=color, marker="o", label=run_name))

        ax.set_title(DATASET_LABELS.get(dataset, dataset), loc="left", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(_ylabel(metric))
        ax.set_ylim(0, 1.15)
        ax.grid(axis="y", zorder=0)

    axes[-1, 0].set_xlabel(xlabel)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    handles = legend_handles[:]
    if metric != "error":
        handles.append(Line2D([], [], linestyle="none", label=f"metric: {_metric_title(metric)} per bin"))
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(3, len(handles)),
        frameon=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {output}")


def _parse_distance_bins(value: str) -> list[tuple[int, int]]:
    bins: list[tuple[int, int]] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
        else:
            lo = hi = int(part)
        if lo < 0 or hi < lo:
            raise ValueError(f"Invalid distance bin: {part!r}")
        bins.append((lo, hi))
    if not bins:
        raise ValueError("--distance-bins did not contain any bins")
    return bins


def _balanced_bins(records_by_run: dict[str, list[dict]], n_bins: int) -> list[tuple[int, int]]:
    if n_bins <= 0:
        raise ValueError("--balanced-bins must be positive")
    distances = sorted({
        int(r["offset_abs"])
        for records in records_by_run.values()
        for r in records
        if int(r["offset_abs"]) > 0
    })
    if not distances:
        return []
    n_bins = min(n_bins, len(distances))
    bins: list[tuple[int, int]] = []
    prev_end_idx = -1
    for i in range(n_bins):
        end_idx = round((i + 1) * len(distances) / n_bins) - 1
        if end_idx <= prev_end_idx:
            continue
        lo = distances[prev_end_idx + 1]
        hi = distances[end_idx]
        bins.append((lo, hi))
        prev_end_idx = end_idx
    return bins


def _range_label(lo: int, hi: int) -> str:
    return str(lo) if lo == hi else f"{lo}-{hi}"


def _make_plot_bins(ranges: list[tuple[int, int]], symmetric: bool, include_zero: bool) -> list[dict]:
    bins: list[dict] = []
    if symmetric:
        for lo, hi in reversed(ranges):
            label = _range_label(lo, hi)
            bins.append({
                "label": f"-{label}",
                "lo": lo,
                "hi": hi,
                "side": "neg",
                "offset_steps": -lo,
            })
        if include_zero:
            bins.append({"label": "0", "lo": 0, "hi": 0, "side": "zero", "offset_steps": 0})
        for lo, hi in ranges:
            label = _range_label(lo, hi)
            bins.append({
                "label": f"+{label}",
                "lo": lo,
                "hi": hi,
                "side": "pos",
                "offset_steps": lo,
            })
        return bins

    if include_zero:
        bins.append({"label": "0", "lo": 0, "hi": 0, "side": "zero", "offset_steps": 0})
    for lo, hi in ranges:
        label = _range_label(lo, hi)
        bins.append({"label": label, "lo": lo, "hi": hi, "side": "abs", "offset_steps": lo})
    return bins


def _metric_title(metric: str) -> str:
    return "error" if metric == "error" else metric.upper() if metric == "f1" else metric


def _ylabel(metric: str) -> str:
    if metric == "error":
        return "Error rate (FN at 0, FP elsewhere)"
    return metric.capitalize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", nargs="+", type=Path, required=True, help="Prediction CSV(s) with label, prediction, frame_path")
    parser.add_argument("--run-names", nargs="+", default=None,
                        help="Optional display names for CSVs. Must match --csv length.")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASET_SUBDIRS), default=list(DATASET_SUBDIRS))
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[3] / "data")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[3] / "results" / "evaluation" / "temporal_offsets")
    parser.add_argument("--max-offset", type=int, default=4,
                        help="Maximum absolute offset for signed/absolute plots. Use 0 with binned modes to load all inferred offsets.")
    parser.add_argument("--distance-bins", default=None,
                        help=f"Optional absolute-distance bins, e.g. {DEFAULT_DISTANCE_BINS}. Adds a binned plot and CSV.")
    parser.add_argument("--balanced-bins", type=int, default=None,
                        help="Optional number of roughly balanced absolute-distance bins. Overrides --distance-bins when set.")
    parser.add_argument("--combined-bins", action="store_true",
                        help="Combine negative and positive distances in binned plots. Default keeps symmetric signed bins separate.")
    parser.add_argument("--no-zero-bin", action="store_true",
                        help="Do not include offset 0 in binned plots.")
    parser.add_argument("--combined-plot", action="store_true",
                        help="Also write one multi-panel plot across all requested datasets.")
    parser.add_argument(
        "--metric",
        choices=METRICS,
        default="error",
        help="Metric to plot per offset bin. Stats CSV always includes all metrics.",
    )
    args = parser.parse_args()

    for path in args.csv:
        if not path.exists():
            raise FileNotFoundError(path)
    if args.run_names is not None and len(args.run_names) != len(args.csv):
        raise ValueError("--run-names must have the same number of values as --csv")
    run_names = args.run_names or [path.stem for path in args.csv]

    lookups = []
    required = {"label", "prediction", "frame_path"}
    for run_name, path in zip(run_names, args.csv):
        pred_rows = _load_predictions(path)
        columns = set(pred_rows[0]) if pred_rows else set()
        if missing := required - columns:
            raise ValueError(f"{path} missing columns: {missing}")
        primary, fallback = _prediction_lookup(pred_rows)
        lookups.append((run_name, primary, fallback))

    suffix = "errors" if args.metric == "error" else args.metric
    combined_signed: dict[str, dict[str, list[dict]]] = {}
    combined_absolute: dict[str, dict[str, list[dict]]] = {}
    combined_binned: dict[str, dict[str, list[dict]]] = {}
    combined_binned_label: str | None = None

    for dataset in args.datasets:
        anno_dir = args.data_root / DATASET_SUBDIRS[dataset]
        load_max_offset = None if args.balanced_bins or args.distance_bins else args.max_offset
        rows = _load_offsets(dataset, anno_dir, args.split, load_max_offset)
        if not rows:
            print(f"[SKIP] {dataset}: no offset annotation rows found in {anno_dir}")
            continue

        records_by_run: dict[str, list[dict]] = {}
        for run_name, primary, fallback in lookups:
            records = _analysis_df(rows, primary, fallback, run_name)
            if records:
                records_by_run[run_name] = records

        if not records_by_run:
            print(f"[SKIP] {dataset}: no offset rows matched any prediction CSV")
            continue

        standard_records = {
            run_name: [r for r in records if r["offset_abs"] <= args.max_offset]
            for run_name, records in records_by_run.items()
        }
        standard_records = {run_name: records for run_name, records in standard_records.items() if records}
        if standard_records:
            _write_dataset_outputs(standard_records, dataset, args.output_dir, args.max_offset, args.metric)
            if args.combined_plot:
                signed_by_run, absolute_by_run = _standard_stats_by_run(standard_records, dataset, args.max_offset)
                combined_signed[dataset] = signed_by_run
                combined_absolute[dataset] = absolute_by_run

        if args.balanced_bins:
            ranges = _balanced_bins(records_by_run, args.balanced_bins)
            if ranges:
                bins = _make_plot_bins(ranges, symmetric=not args.combined_bins, include_zero=not args.no_zero_bin)
                _write_binned_outputs(records_by_run, dataset, args.output_dir, args.metric, bins, "balanced_binned")
                if args.combined_plot:
                    combined_binned[dataset] = _binned_stats_by_run(records_by_run, dataset, bins)
                    combined_binned_label = "balanced_binned"
            else:
                print(f"[SKIP] {dataset}: no nonzero offsets available for balanced bins")
        elif args.distance_bins:
            ranges = _parse_distance_bins(args.distance_bins)
            bins = _make_plot_bins(ranges, symmetric=not args.combined_bins, include_zero=not args.no_zero_bin)
            _write_binned_outputs(records_by_run, dataset, args.output_dir, args.metric, bins, "binned")
            if args.combined_plot:
                combined_binned[dataset] = _binned_stats_by_run(records_by_run, dataset, bins)
                combined_binned_label = "binned"

    if args.combined_plot:
        _plot_combined_grid(
            combined_signed,
            f"All datasets: {_metric_title(args.metric)} by signed temporal offset",
            "Signed offset steps",
            args.metric,
            args.output_dir / f"combined_signed_offset_{suffix}.png",
        )
        _plot_combined_grid(
            combined_absolute,
            f"All datasets: {_metric_title(args.metric)} by absolute temporal offset",
            "Absolute offset steps",
            args.metric,
            args.output_dir / f"combined_absolute_offset_{suffix}.png",
        )
        if combined_binned and combined_binned_label:
            _plot_combined_grid(
                combined_binned,
                f"All datasets: {_metric_title(args.metric)} by binned touch distance",
                "Distance from touch",
                args.metric,
                args.output_dir / f"combined_{combined_binned_label}_offset_{suffix}.png",
            )


if __name__ == "__main__":
    main()
