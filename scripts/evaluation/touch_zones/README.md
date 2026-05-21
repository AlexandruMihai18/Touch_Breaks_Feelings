# Touch Zones Scripts

Assign each touch event to a cell in an N×N spatial grid and analyse model performance per zone. Run all from `Touch_Breaks_Feelings/`.

**Availability:** works on all datasets (Greatest Hits, EPIC Kitchen, manual annotations).

---

## Step 1 — Annotate

### `annotate_touch_zones.py`

For every touch-positive entry in one or more annotation JSONs, finds the largest connected blob in the touch mask (dropping noise fragments), computes its centroid, and maps it to a grid cell `(x_touch, y_touch)` where both coordinates are in `[0, N)` (0 = left/top, N−1 = right/bottom).

No-touch entries get `x_touch=null, y_touch=null`. Results are written back in-place.

```bash
python scripts/evaluation/touch_zones/annotate_touch_zones.py \
    data/greatest_hits/annotations/train.json \
    data/greatest_hits/annotations/val.json \
    data/epic_kitchen/annotations/train.json \
    data/epic_kitchen/annotations/val.json
```

Key flags:

| Flag | Default | Description |
| --- | --- | --- |
| `--grid N` | 8 | Grid dimension — produces an N×N layout |
| `--min-blob-area PX` | 100 | Minimum blob size in pixels; smaller blobs are treated as noise |
| `--force` | off | Recompute even if `x_touch`/`y_touch` already set |
| `--dry-run` | off | Compute and print results without writing |

**When:** Once before evaluation, or whenever annotation files change. Re-run with `--force` if `--grid` changes.

---

## Step 2 — Visualise distribution (optional)

### `visualize_touch_zones.py`

Plots an N×N heatmap of touch-event counts per grid cell. No predictions needed — useful for understanding dataset bias before running any model.

```bash
python scripts/evaluation/touch_zones/visualize_touch_zones.py \
    data/greatest_hits/annotations/train.json \
    data/epic_kitchen/annotations/train.json \
    --title "All datasets — touch zone distribution" \
    --output results/touch_zone_distribution.png
```

Key flags:

| Flag | Default | Description |
| --- | --- | --- |
| `--grid N` | 8 | Must match the value used in `annotate_touch_zones.py` |
| `--title` | "Touch zone distribution" | Plot title |
| `--output` | `<first_annotation_dir>/touch_zone_distribution.png` | Output PNG path |

**When:** After `annotate_touch_zones.py`. Re-run whenever the dataset composition changes.

---

## Step 3 — Analyse model performance

### `analyze_grid_performance.py`

Reads a predictions CSV, groups touch-positive samples by their `(x_touch, y_touch)` cell, and renders an N×N heatmap coloured by recall (hit rate) per zone. Gray cells have no samples.

Also prints global accuracy, F1, and a top/bottom-5 zone summary to stdout.

```bash
python scripts/evaluation/touch_zones/analyze_grid_performance.py \
    results/predictions.csv \
    --metric recall \
    --output results/grid_performance.png
```

Key flags:

| Flag | Default | Description |
| --- | --- | --- |
| `--metric` | recall | `recall` or `accuracy` (equivalent for touch-only samples per zone) |
| `--grid N` | 8 | Must match the value used in `annotate_touch_zones.py` |
| `--output` | `<csv_stem>_heatmap.png` | Output PNG path |

**Note on F1:** F1 per zone is undefined because false positives come from no-touch samples which have no spatial location. Per-zone recall (what fraction of touch events in each zone were detected) is the meaningful metric here. Global F1 is printed as an overall summary.

**When:** After the model has produced a predictions CSV that includes `x_touch` and `y_touch` columns joined from the annotation JSONs.

---

## Typical run order

```text
annotate_touch_zones.py           ← adds x_touch, y_touch to JSONs (one-time setup)
        ↓
visualize_touch_zones.py          ← optional: inspect spatial bias in the dataset
        ↓
[model training + evaluation]
        ↓
analyze_grid_performance.py       ← spatial heatmap of recall per zone
```
