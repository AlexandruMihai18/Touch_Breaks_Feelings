# Greatest Hits SLURM Jobs

Four jobs covering the full Greatest Hits pipeline. Submit with `sbatch` from `touch_from_segmentation/`.

See [`scripts/greatest_hits/README.md`](../../scripts/greatest_hits/README.md) for the underlying Python scripts and their full argument reference.

---

## Jobs

### `download.sh`
Downloads and extracts the raw dataset (videos + `*_times.txt` labels). Kept separate because it has a different resource profile — pure network I/O, no Python env needed.

```bash
sbatch jobs/greatest_hits/download.sh
```

| Resource | Value |
|---|---|
| Partition | `staging` |
| CPUs | 2 |
| Memory | 4 GB |
| Time limit | 6 h |

**Notes:** Downloads the 456×256 low-resolution version by default (~20 GB). Edit `--low` to `--full` for full resolution (~50 GB).

---

### `prepare_data.sh`
Runs the two lightweight CPU steps sequentially: frame extraction then GT annotation generation.

```bash
sbatch jobs/greatest_hits/prepare_data.sh
```

| Resource | Value |
|---|---|
| Partition | `staging` |
| CPUs | 16 |
| Memory | 16 GB |
| Time limit | 2 h |

**Steps inside:**
1. `2_extract_greatest_hits_frames.py` — decodes annotated frames from each video (~1h, uses all 16 CPUs)
2. `3_generate_gh_gt_annotations.py` — writes `train.json`, `val.json`, and `metadata.csv` from `*_times.txt` labels (~few minutes)

**Depends on:** `download.sh`

---

### `annotate.sh`
Runs the full per-frame annotation pipeline: Grounding-DINO → SAM → Depth-Anything-V2 → touch mask. GPU required. Uses a 4-task SLURM array to split the video folder list round-robin across tasks.

```bash
sbatch jobs/greatest_hits/annotate.sh
```

| Resource | Value |
|---|---|
| Partition | `gpu_mig` |
| GPUs | 1 per task |
| CPUs | 8 per task |
| Memory | 32 GB per task |
| Time limit | 8 h |
| Array | 4 tasks |

**Notes:** Uses `--skip-existing` by default — safe to resubmit to resume an interrupted run. Update `--array` and `NUM_JOBS` together to change the degree of parallelism.

**Depends on:** `prepare_data.sh`

---

### `finalize.sh`
Runs the two lightweight CPU steps after annotation: enriching annotations with mask paths and evaluating the pipeline.

```bash
sbatch jobs/greatest_hits/finalize.sh
```

| Resource | Value |
|---|---|
| Partition | `staging` |
| CPUs | 4 |
| Memory | 8 GB |
| Time limit | 1 h |

**Steps inside:**
1. `5_generate_gh_mask_annotations.py` — adds `stick_mask_path`, `object_mask_path`, `target_path` to the annotation JSONs and builds `*_ctx_index.json`
2. `6_evaluate_annotation_pipeline.py` — writes `results/gh_eval.csv` with per-split and per-material precision/recall/F1

**Depends on:** `annotate.sh`

---

## Run order

```text
download.sh
    ↓
prepare_data.sh
    ↓
annotate.sh        ← GPU partition, long
    ↓
finalize.sh
```
