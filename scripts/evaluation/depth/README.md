# Depth Evaluation Scripts

Assign each touch event a depth proxy value and analyse model performance across depth bins. Run all from `Touch_Breaks_Feelings/`.

**Availability:** Greatest Hits only — depth maps (`*_depth.png`) are produced by the GH annotation pipeline. EPIC Kitchen entries will have `depth_touch=null` until depth maps are generated.

**Depth convention:** depth maps are RGB PNGs colorized with the **inferno** colormap over a Depth-Anything-V2 disparity output:

```
dark / purple  (low grayscale value)  →  far from camera
bright / yellow (high grayscale value) →  near / close to camera
```

---

## Step 1 — Annotate

### `annotate_touch_depth.py`

For every touch-positive entry, resolves the depth file at `{image_stem}_depth.png` in the same directory as the image, converts it to grayscale, and stores the median value under the touch mask as `depth_touch` (float in [0, 255]).

No-touch entries and entries without a depth file get `depth_touch=null`. Results are written back in-place.

```bash
python scripts/evaluation/depth/annotate_touch_depth.py \
    data/greatest_hits/annotations/train.json \
    data/greatest_hits/annotations/val.json
```

Key flags:

| Flag | Default | Description |
| --- | --- | --- |
| `--force` | off | Recompute even if `depth_touch` already set |
| `--dry-run` | off | Compute and print results without writing |

**When:** Once before evaluation, or whenever the touch masks change. EK entries will report `no_depth` — this is expected until depth maps are generated for that dataset.

---

## Step 2 — Visualise distribution (optional)

### `visualize_depth_distribution.py`

Plots a histogram of `depth_touch` values across all annotation files. No predictions needed — useful for understanding depth bias in the dataset before running any model.

```bash
python scripts/evaluation/depth/visualize_depth_distribution.py \
    data/greatest_hits/annotations/train.json \
    data/greatest_hits/annotations/val.json \
    --title "Greatest Hits — touch depth distribution" \
    --output results/depth_distribution.png
```

Key flags:

| Flag | Default | Description |
| --- | --- | --- |
| `--bins N` | 20 | Number of histogram bins |
| `--title` | "Touch depth distribution" | Plot title |
| `--output` | `<first_annotation_dir>/depth_distribution.png` | Output PNG path |

**When:** After `annotate_touch_depth.py`. Re-run whenever the dataset changes.

---

## Step 3 — Analyse model performance

### `analyze_depth_performance.py`

Reads a predictions CSV, bins touch-positive samples by `depth_touch` using quantiles (equal samples per bin), and renders a bar chart of recall (hit rate) per bin coloured by performance.

Also prints per-bin stats and global accuracy/F1 to stdout.

```bash
python scripts/evaluation/depth/analyze_depth_performance.py \
    results/predictions.csv \
    --n-bins 5 \
    --output results/depth_performance.png
```

Key flags:

| Flag | Default | Description |
| --- | --- | --- |
| `--n-bins N` | 5 | Number of quantile depth bins |
| `--output` | `<csv_stem>_depth_perf.png` | Output PNG path |

**Note on bins:** quantile binning ensures each bar has roughly the same number of samples. Reduce `--n-bins` if some bins end up with very few samples (< 5).

**When:** After the model has produced a predictions CSV that includes the `depth_touch` column joined from the annotation JSONs.

---

## Typical run order

```text
annotate_touch_depth.py           ← adds depth_touch to JSONs (one-time setup)
        ↓
visualize_depth_distribution.py   ← optional: inspect depth bias in the dataset
        ↓
[model training + evaluation]
        ↓
analyze_depth_performance.py      ← bar chart of recall per depth bin
```
