# EPIC Kitchen SLURM Jobs

SLURM job scripts for the EPIC Kitchen pipeline. Submit with `sbatch` from `Touch_Breaks_Feelings/`.

See [`scripts/epic_kitchen/README.md`](../../scripts/epic_kitchen/README.md) for the underlying Python scripts and their full argument reference.

---

## Jobs

### `download_epic_kitchen.sh`

Downloads VISOR annotations, extracts frames, renders mask triplets, stream-extracts audio from
EK100 videos, generates annotation JSONs, and runs a sanity-check — all participants in one job.

```bash
sbatch jobs/epic_kitchen/download_epic_kitchen.sh
```

| Resource | Value |
| --- | --- |
| Partition | `staging` (no GPU) |
| CPUs | 32 |
| Memory | 32 GB |
| Time limit | 8 h |

**Notes:** `--workers` is auto-set to `SLURM_CPUS_PER_TASK`. Downloads train split only; edit
`--split` for `val`. Loads `FFmpeg/6.0-GCCcore-12.3.0` for Phase 6 audio extraction — verify the
module name matches what is available on the cluster (`module avail FFmpeg`). Add `--skip-audio`
to the python command to skip audio extraction and restore the original 4 h runtime.

---

### `refine_touch_masks.sh`

Runs Depth-Anything-V2 + guided filter on every annotated frame to produce `*_touch_refined.png` masks.

```bash
sbatch jobs/epic_kitchen/refine_touch_masks.sh
```

| Resource | Value |
| --- | --- |
| Partition | `gpu_mig` |
| GPUs | 1 |
| CPUs | 4 |
| Time limit | 8 h |

**Notes:** Processes both train and val splits. Add `--overwrite` inside the script to regenerate
existing refined masks. Depends on `download_epic_kitchen.sh` having completed first.

---

### `dilate_touch_masks.sh`

Generates constrained-dilated mask variants (`*_dilated_r40.png`) for both the original and
depth-refined touch masks.

```bash
sbatch jobs/epic_kitchen/dilate_touch_masks.sh
```

| Resource | Value |
| --- | --- |
| Partition | `staging` (no GPU) |
| CPUs | 16 |
| Memory | 16 GB |
| Time limit | 30 min |

**Notes:** Runs two passes — once on `*_touch.png` and once on `*_touch_refined.png`. The refined
pass requires `refine_touch_masks.sh` to have completed. Adjust `--radii` in the script to explore
different dilation sizes.

---

### `evaluate_refined_masks.sh`

Evaluates depth-refined masks as binary classification (touch / no-touch) and writes results to
`results/ek_refined_eval.csv`.

```bash
sbatch jobs/epic_kitchen/evaluate_refined_masks.sh
```

| Resource | Value |
| --- | --- |
| Partition | `staging` (no GPU) |
| CPUs | 4 |
| Memory | 8 GB |
| Time limit | 30 min |

**Notes:** Requires `refine_touch_masks.sh` to have completed. Outputs a per-frame CSV and a
per-object-category summary CSV.

---

### `fix_annotation_paths.sh`

Rewrites path prefixes inside every JSON under the annotations directory after
files have been moved to scratch. Uses `sed` in-place — fast and safe for
simple prefix swaps.

```bash
sbatch jobs/epic_kitchen/fix_annotation_paths.sh
```

| Resource | Value |
| --- | --- |
| Partition | `staging` (no GPU) |
| CPUs | 1 |
| Memory | 4 GB |
| Time limit | 10 min |

**Notes:** Edit `OLD_PREFIX` and `NEW_PREFIX` inside the script if the source or
destination paths differ. Typically run once after `copy_job.sh` completes and
before any training job that reads the annotations. Idempotent — safe to re-run
if the prefix is already correct (no-op).

---

### `repair_depth_masks.sh`

Recomputes depth PNGs and refined touch masks for entries that are missing or
zero-byte (e.g. after a disk-full failure during `refine_touch_masks.sh`).
Skips entries that already have both valid files — safe to re-run.

```bash
sbatch jobs/epic_kitchen/repair_depth_masks.sh
```

| Resource | Value |
| --- | --- |
| Partition | `gpu_a100` |
| GPUs | 1 |
| CPUs | 4 |
| Time limit | 8 h |

**Notes:** Reads annotations from `/scratch-shared/<user>/epic_kitchen/annotations`.
Uses a 4-task SLURM array; update `--array` and `NUM_JOBS` together to change
parallelism. Workers do not write back to the annotation JSON (to avoid
concurrent writes) — run `refine_touch_masks.sh` without `--overwrite`
afterwards to backfill the `depth_path` field. Add `--dry-run` inside the
script to preview what would be reprocessed without writing anything. Run after
`refine_touch_masks.sh` has completed (or failed partway).

---

## Typical run order

```text
download_epic_kitchen.sh
        ↓
copy_job.sh  →  fix_annotation_paths.sh   (run after copy, before training)
        ↓
refine_touch_masks.sh
        ↓
dilate_touch_masks.sh   evaluate_refined_masks.sh  (can run in parallel)
```
