# Object Coverage Scripts

Annotate every sample with the fraction of the image covered by its object mask, then analyse model performance across coverage bins. Run all from `Touch_Breaks_Feelings/`.

**Availability:** works on both Greatest Hits and EPIC Kitchen — all entries in both datasets have an `object_mask_path`.

**Key difference from depth and zone analyses:** `object_coverage` is available for **all** samples (touch and no-touch), so the performance analysis computes full F1 and accuracy per bin, not just per-zone recall.

---

## Step 1 — Annotate

### `annotate_object_coverage.py`

For every entry with an `object_mask_path`, computes:

    object_coverage = nonzero pixels / total pixels  ∈ [0, 1]

Results are written back in-place. Entries without a mask path or with a missing file get `object_coverage=null`.

```bash
python scripts/evaluation/object_coverage/annotate_object_coverage.py \
    data/greatest_hits/annotations/train.json \
    data/greatest_hits/annotations/val.json \
    data/epic_kitchen/annotations/train.json \
    data/epic_kitchen/annotations/val.json
```

Key flags:

| Flag | Default | Description |
| --- | --- | --- |
| `--force` | off | Recompute even if `object_coverage` already set |
| `--dry-run` | off | Compute and print results without writing |

**When:** Once before evaluation, or after annotation files are regenerated. Safe to re-run.

---

## Step 2 — Visualise distribution (optional)

### `visualize_object_coverage.py`

Plots overlaid histograms of `object_coverage` for touch-positive and no-touch samples. Useful for detecting dataset bias — if no-touch samples systematically have higher or lower coverage than touch samples, coverage correlates with the label and the model may be exploiting it.

```bash
python scripts/evaluation/object_coverage/visualize_object_coverage.py \
    data/greatest_hits/annotations/train.json \
    data/epic_kitchen/annotations/train.json \
    --title "All datasets — object mask coverage" \
    --output results/coverage_distribution.png
```

Key flags:

| Flag | Default | Description |
| --- | --- | --- |
| `--bins N` | 25 | Number of histogram bins |
| `--title` | "Object mask coverage distribution" | Plot title |
| `--output` | `<first_annotation_dir>/object_coverage_distribution.png` | Output PNG path |

**When:** After `annotate_object_coverage.py`. Re-run whenever the dataset changes.

---

## Step 3 — Analyse model performance

### `analyze_coverage_performance.py`

Reads a predictions CSV, bins all samples by `object_coverage` using quantiles (equal samples per bin), and renders a bar chart of F1 or accuracy per bin coloured by performance.

Because coverage is available for all samples, each bin contains a mix of touch and no-touch entries — enabling full F1 and accuracy metrics, not just recall.

```bash
python scripts/evaluation/object_coverage/analyze_coverage_performance.py \
    results/predictions.csv \
    --metric f1 \
    --output results/coverage_performance.png
```

Key flags:

| Flag | Default | Description |
| --- | --- | --- |
| `--n-bins N` | 5 | Number of quantile coverage bins |
| `--metric` | f1 | `f1` or `accuracy` |
| `--output` | `<csv_stem>_coverage_perf.png` | Output PNG path |

**When:** After the model has produced a predictions CSV with an `object_coverage` column joined from the annotation JSONs.

---

## Typical run order

```text
annotate_object_coverage.py           ← adds object_coverage to JSONs (one-time setup)
        ↓
visualize_object_coverage.py          ← optional: inspect coverage bias in the dataset
        ↓
[model training + evaluation]
        ↓
analyze_coverage_performance.py       ← bar chart of F1/accuracy per coverage bin
```
