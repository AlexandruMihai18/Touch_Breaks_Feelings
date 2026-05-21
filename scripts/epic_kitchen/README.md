# EPIC Kitchen Scripts

Scripts for building and maintaining the EPIC Kitchen touch-mask dataset. Run all from `touch_from_segmentation/`.

---

## Data pipeline (run in order)

### `download_epic_kitchen/`

Downloads VISOR annotations, extracts annotated frames, renders hand/object/touch mask triplets,
generates `train.json` / `val.json` annotation files, and runs a sanity-check — all in a single
pass (phases 1–5 + verify).

```bash
python scripts/epic_kitchen/download_epic_kitchen \
    --all-participants \
    --match-no-contact \
    --split train \
    --workers 8
```

Add `--skip-verify` to suppress the post-download readiness check.

**When:** First-time setup. Produces everything under `data/epic_kitchen/` including the final annotation JSONs.

---

### `refine_touch_masks.py`  *(GPU)*

Runs Depth-Anything-V2 + optional guided image filter on every annotated frame and writes
`*_touch_refined.png` beside the original touch mask. The original is never modified.

```bash
python scripts/epic_kitchen/refine_touch_masks.py
```

**When:** After `download_epic_kitchen`. GPU recommended; slow per-frame.

---

## Analysis (run independently)

### `evaluate_refined_masks.py`

Treats refined masks as binary predictions and computes precision/recall/F1 against the original
GT labels, broken down by split and object category.

```bash
python scripts/epic_kitchen/evaluate_refined_masks.py \
    --output results/ek_refined_eval.csv
```

**When:** After `refine_touch_masks.py`.

---

### `dilate_touch_masks.py`

Generates dilated mask variants (e.g. `*_touch_dilated_r40.png`) constrained to the union of
hand + object pixels. Can operate on original or refined masks. Use to explore different dilation
budgets without overwriting anything.

```bash
python scripts/epic_kitchen/dilate_touch_masks.py --radii 15 20 30
python scripts/epic_kitchen/dilate_touch_masks.py --input-suffix touch_refined --radii 20
```

**When:** Ablations or experiments comparing different dilation sizes.

---

### `compute_object_coverage.py`

Computes median object mask coverage (nonzero pixels / total pixels) per category. Use to decide
which object categories to keep for training.

```bash
python scripts/epic_kitchen/compute_object_coverage.py --min-coverage 0.005
```

**When:** To audit or update the object label filter.

---

## Unified runner

### `run_pipeline.py`

Calls all pipeline steps sequentially. Use `--steps` to run a subset or `--skip` to skip individual steps.

```bash
# Full pipeline
python scripts/epic_kitchen/run_pipeline.py

# Skip download (data already on disk)
python scripts/epic_kitchen/run_pipeline.py --skip download

# Preview commands without running
python scripts/epic_kitchen/run_pipeline.py --dry-run
```

---

## Typical run order

```text
download_epic_kitchen/              ← frames + masks + annotations (CPU, ~hours)
        ↓
refine_touch_masks.py               ← depth-filtered masks (GPU, slow)
        ↓
evaluate_refined_masks.py           ← optional, analysis only
```
