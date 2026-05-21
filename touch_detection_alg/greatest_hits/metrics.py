"""Evaluation helpers — compare annotation pipeline predictions against GH ground truth."""

import json
from pathlib import Path


def compute_gh_metrics(video_id: str, dataset_rows: list[dict], annotations_dir: Path) -> str:
    """Compare model predictions against Greatest Hits ground truth from annotation JSONs.

    Reads the GT touch/no-touch labels from train.json + val.json (produced by
    3_generate_gh_gt_annotations.py) and matches them against the `type` field
    in dataset_rows (set by the annotation pipeline).

    Returns a human-readable string with TP/FP/FN/TN and precision/recall/F1.
    """
    gt: dict[int, tuple[str, str | None]] = {}
    for split in ("train", "val"):
        path = annotations_dir / f"{split}.json"
        if not path.exists():
            continue
        for entry in json.loads(path.read_text()):
            if entry.get("video_id") != video_id:
                continue
            frame_idx = entry.get("frame_idx")
            if frame_idx is not None:
                gt[int(frame_idx)] = (entry["type"], entry.get("action"))

    if not gt:
        return f"\n(No GT entries for {video_id} — run 3_generate_gh_gt_annotations.py)"

    pred = {
        int(Path(row["image_path"]).stem.split("_")[1]): row["type"]
        for row in dataset_rows
    }

    matched = [(gt[idx][0], gt[idx][1], pred[idx]) for idx in gt if idx in pred]
    if not matched:
        return "\n(No frame overlap between GT annotations and propagated frames)"

    tp = sum(1 for g, _, p in matched if g == "touch"    and p == "touch")
    fp = sum(1 for g, _, p in matched if g == "no-touch" and p == "touch")
    fn = sum(1 for g, _, p in matched if g == "touch"    and p == "no-touch")
    tn = sum(1 for g, _, p in matched if g == "no-touch" and p == "no-touch")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    lines = [
        f"\n── Greatest Hits metrics ({len(matched)} frames) ──",
        f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}",
        f"  Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}",
    ]

    action_groups: dict[str, list[tuple[str, str]]] = {}
    for g, action, p in matched:
        key = action or "(unspecified)" if g == "touch" else "(no-touch)"
        action_groups.setdefault(key, []).append((g, p))

    if len(action_groups) > 1:
        lines.append("  Per action:")
        for action_key in sorted(action_groups):
            pairs = action_groups[action_key]
            atp = sum(1 for g, p in pairs if g == "touch"    and p == "touch")
            afp = sum(1 for g, p in pairs if g == "no-touch" and p == "touch")
            afn = sum(1 for g, p in pairs if g == "touch"    and p == "no-touch")
            af1 = 2 * atp / (2 * atp + afp + afn) if (2 * atp + afp + afn) > 0 else 0.0
            lines.append(f"    {action_key:<16} TP={atp} FP={afp} FN={afn}  F1: {af1:.3f}")

    return "\n".join(lines)
