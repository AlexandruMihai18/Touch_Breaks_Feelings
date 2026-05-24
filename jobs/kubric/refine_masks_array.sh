#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --job-name=kubric_refine_masks
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-task=1
#SBATCH --time=8:00:00
#SBATCH --output=slurm_output_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=daniel.otero.gomez@student.uva.nl

# Array job: each task processes a round-robin shard of Kubric touch entries,
# writing three files per entry alongside existing masks:
#   *_touch_refined_gt.png    — touch mask refined with GT depth
#   *_touch_refined_pred.png  — touch mask refined with Depth-Anything-V2
#   *_depth_pred.png          — predicted depth uint16 PNG
# Entries that already have all three valid files are skipped (safe to re-run).
#
# JSON write-back is skipped during the array run to avoid concurrent overwrites.
# After all tasks finish, run a single sequential pass to flush the new fields
# (touch_refined_gt_path, touch_refined_pred_path, depth_pred_path) into JSON:
#
#   python scripts/kubric/refine_touch_masks.py \
#       --anno-dir "/scratch-shared/$USER/kubric_movi_a_256/annotations" \
#       --splits val
#
# Submit:
#   sbatch --array=0-7 jobs/kubric/refine_masks_array.sh

N_JOBS=8
ANNO_DIR="/scratch-shared/$(whoami)/kubric_movi_a_256/annotations"

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate touch_from_segmentation

cd "$HOME/Touch_Breaks_Feelings"

python scripts/kubric/refine_touch_masks.py \
    --anno-dir  "${ANNO_DIR}" \
    --splits    val \
    --num-jobs  "${N_JOBS}" \
    --job-index "${SLURM_ARRAY_TASK_ID}"
