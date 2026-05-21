# EPIC Kitchen SLURM Jobs

SLURM job scripts for the EPIC Kitchen pipeline. Submit with `sbatch` from `touch_from_segmentation/`.

See [`scripts/epic_kitchen/README.md`](../../scripts/epic_kitchen/README.md) for the underlying Python scripts and their full argument reference.

---

## Jobs

### `download_epic_kitchen.sh`
Downloads VISOR annotations, extracts frames, renders mask triplets, generates annotation JSONs, and runs a sanity-check — all participants in one job.

```bash
sbatch jobs/epic_kitchen/download_epic_kitchen.sh
```

| Resource | Value |
|---|---|
| Partition | `staging` (no GPU) |
| CPUs | 32 |
| Memory | 32 GB |
| Time limit | 4 h |

**Notes:** `--workers` is auto-set to `SLURM_CPUS_PER_TASK`. Downloads train split only; edit `--split` for `val`. Add `--skip-verify` to suppress the post-download readiness check.

---

### `refine_touch_masks.sh`
Runs Depth-Anything-V2 + guided filter on every annotated frame to produce `*_touch_refined.png` masks.

```bash
sbatch jobs/epic_kitchen/refine_touch_masks.sh
```

| Resource | Value |
|---|---|
| Partition | `gpu_mig` |
| GPUs | 1 |
| CPUs | 4 |
| Time limit | 8 h |

**Notes:** Processes both train and val splits. Add `--overwrite` inside the script to regenerate existing refined masks. Depends on `download_epic_kitchen.sh` having completed first.

---

### `dilate_touch_masks.sh`
Generates constrained-dilated mask variants (`*_dilated_r40.png`) for both the original and depth-refined touch masks.

```bash
sbatch jobs/epic_kitchen/dilate_touch_masks.sh
```

| Resource | Value |
|---|---|
| Partition | `staging` (no GPU) |
| CPUs | 16 |
| Memory | 16 GB |
| Time limit | 30 min |

**Notes:** Runs two passes — once on `*_touch.png` and once on `*_touch_refined.png`. The refined pass requires `refine_touch_masks.sh` to have completed. Adjust `--radii` in the script to explore different dilation sizes.

---

### `evaluate_refined_masks.sh`
Evaluates depth-refined masks as binary classification (touch / no-touch) and writes results to `results/ek_refined_eval.csv`.

```bash
sbatch jobs/epic_kitchen/evaluate_refined_masks.sh
```

| Resource | Value |
|---|---|
| Partition | `staging` (no GPU) |
| CPUs | 4 |
| Memory | 8 GB |
| Time limit | 30 min |

**Notes:** Requires `refine_touch_masks.sh` to have completed. Outputs a per-frame CSV and a per-object-category summary CSV.

---

## Typical run order

```
download_epic_kitchen.sh
        ↓
refine_touch_masks.sh
        ↓
dilate_touch_masks.sh   evaluate_refined_masks.sh  (can run in parallel)
```
