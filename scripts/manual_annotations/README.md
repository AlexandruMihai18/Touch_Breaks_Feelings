# Manual Annotations Scripts

Scripts for preparing videos for annotation and managing the outputs of the manual annotation app. Run all from `touch_from_segmentation/`.

These scripts handle everything around that app: preparing input frames before annotation, and cleaning up / aggregating outputs afterwards. The app produces, per video session: `manifest.json` (annotated frame list), `dataset.json` (per-frame mask paths and labels), and the mask PNGs (`_touch`, `_object`, `_hand`, `_depth`) under `masks/{video_id}/`.

---

## Pre-annotation

### `preprocess_frames.py`

Extracts N pseudo-uniformly sampled frames per video from a raw video directory and writes them as JPEGs. Run this before opening a video in the annotation app to get a diverse spread of frames without dealing with the full video.

```bash
python scripts/manual_annotations/preprocess_frames.py \
    --video_dir data/raw_videos \
    --output_dir data/manual_annotations/frames \
    --n_frames 20 \
    --num_workers 16
```

SLURM job-array aware: pass `--num_jobs $SLURM_ARRAY_TASK_COUNT` and `--job_id $SLURM_ARRAY_TASK_ID` to shard across array tasks.

**When:** Once per new batch of raw videos, before annotation begins. Safe to resume — skips already-extracted frames by default.

---

## Post-annotation

### `cleanup_frames.py`

Prunes an annotation session: keeps the N most-recently-modified frames per video folder and removes everything else — the frame JPEG, its mask files (`_touch`, `_object`, `_hand`, `_depth`), and the corresponding entries in `manifest.json` and `dataset.json`. Also deletes `annotations/train.json` and `annotations/val.json` since they become stale.

```bash
# Always preview first
python scripts/manual_annotations/cleanup_frames.py --dry-run

# Execute
python scripts/manual_annotations/cleanup_frames.py --keep 20
```

**When:** After an annotation session produces more frames than needed. Run `generate_annotations.py` afterwards to rebuild the train/val splits.

---

### `merge_datasets.py`

Merges all `dataset.json` files found in subdirectories of `data/manual_annotations/annotations/` into a single `merged_dataset.json`. Useful when annotations are spread across multiple sessions or annotators.

```bash
# Default output: data/manual_annotations/annotations/merged_dataset.json
python scripts/manual_annotations/merge_datasets.py

# Custom output path
python scripts/manual_annotations/merge_datasets.py /path/to/output.json
```

**When:** Before training, once all annotation sessions are complete and you need a single flat file spanning all sessions.

---

## Typical workflow

```text
[raw videos]
      ↓
preprocess_frames.py          ← extract diverse frames before annotation
      ↓
[annotation app]              ← produces manifest.json, dataset.json, masks
      ↓
cleanup_frames.py             ← optional: prune excess or low-quality frames
      ↓
merge_datasets.py             ← combine multiple sessions into one flat file
      ↓
generate_annotations.py       ← build train/val splits (scripts/epic_kitchen/)
```
