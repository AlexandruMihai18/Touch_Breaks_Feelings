# Greatest Hits Scripts

Scripts for building and maintaining the Greatest Hits touch-mask dataset. Run all from `touch_from_segmentation/`.

---

## Data pipeline (run in order)

### `1_download_greatest_hits.sh`

Downloads and extracts the raw Greatest Hits data (videos + `*_times.txt` label files).

```bash
bash scripts/greatest_hits/1_download_greatest_hits.sh --low     # recommended (~20 GB)
bash scripts/greatest_hits/1_download_greatest_hits.sh --full    # full resolution (~50 GB)
bash scripts/greatest_hits/1_download_greatest_hits.sh --features  # precomputed sound features (~1 GB)
```

**When:** First-time setup only. Validates existing zips before re-downloading.

---

### `2_extract_greatest_hits_frames.py`

Reads each `*_denoised.mp4` + `*_times.txt` pair and extracts the annotated contact frames as JPEGs. SLURM job-array aware for parallelism.

```bash
python scripts/greatest_hits/2_extract_greatest_hits_frames.py \
    --data_dir   data/greatest_hits/vis-data-256/vis-data-256 \
    --output_dir data/greatest_hits/frames \
    --num_workers 8
```

**When:** After `1_download_greatest_hits.sh`. Safe to resume — skips already-extracted frames by default.

---

### `3_generate_gh_gt_annotations.py`

Reads `*_times.txt` labels and extracted frame paths in a single pass and writes:

- `annotations/train.json` + `annotations/val.json` — with `type: touch/no-touch`, no mask paths yet
- `frames/metadata.csv` — flat per-frame table with all label fields (material / action / reaction), used as GT by `5_evaluate_annotation_pipeline.py`

```bash
python scripts/greatest_hits/3_generate_gh_gt_annotations.py
```

**When:** Right after frame extraction. Gives you class balance and evaluation GT before the slow annotation step.

---

### `4_annotate_greatest_hits.py`

Runs the full annotation pipeline (Grounding-DINO → SAM → Depth-Anything-V2 → guided filter → `compute_touch_region_v2`) on every extracted frame. Writes stick/object/touch masks per video.

```bash
# All videos
python scripts/greatest_hits/4_annotate_greatest_hits.py --skip-existing

# Single video
python scripts/greatest_hits/4_annotate_greatest_hits.py --video-id 2015-03-28-19-34-13_denoised
```

**When:** After frame extraction. GPU recommended. This is the slow step — use `--skip-existing` to resume.

---

### `5_generate_gh_mask_annotations.py`

Enriches the existing `train.json` / `val.json` with mask paths (`stick_mask_path`, `object_mask_path`, `target_path`) resolved from the masks directory. Also builds `*_ctx_index.json` for context sampling at training time.

```bash
python scripts/greatest_hits/5_generate_gh_mask_annotations.py
```

**When:** After `4_annotate_greatest_hits.py`. Re-run whenever new videos are annotated.

---

## Analysis

### `6_evaluate_annotation_pipeline.py`

Evaluates the annotation pipeline as binary classification (touch / no-touch). GT comes from `metadata.csv`; predictions come from the per-video masks. Reports per-split and per-material precision/recall/F1.

```bash
python scripts/greatest_hits/6_evaluate_annotation_pipeline.py \
    --output results/gh_eval.csv
```

**When:** After both `3_generate_gh_gt_annotations.py` and `4_annotate_greatest_hits.py` have completed.

---

## Unified runner

### `run_pipeline.py`

Calls all steps sequentially. Use `--steps` to run a subset or `--skip` to skip individual steps.

```bash
# Full pipeline
python scripts/greatest_hits/run_pipeline.py --download-mode low

# Skip download (data already on disk)
python scripts/greatest_hits/run_pipeline.py --skip download

# Only annotation steps
python scripts/greatest_hits/run_pipeline.py --steps gt-annotations annotate mask-annotations

# Preview commands without running
python scripts/greatest_hits/run_pipeline.py --dry-run
```

---

## Typical run order

```text
1_download_greatest_hits.sh
        ↓
2_extract_greatest_hits_frames.py
        ↓
3_generate_gh_gt_annotations.py   ← writes metadata.csv + train/val JSON (GT labels, no masks)
        ↓
4_annotate_greatest_hits.py       ← GPU, slow
        ↓
5_generate_gh_mask_annotations.py ← enriches annotations with mask paths + builds ctx_index
        ↓
6_evaluate_annotation_pipeline.py ← optional, analysis only
```
