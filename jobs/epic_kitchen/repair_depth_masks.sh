#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=repair_depth
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --array=0-3                          # 4 parallel tasks (indices 0-3)
#SBATCH --output=slurm_output_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=daniel.otero.gomez@student.uva.nl

# Parallelised depth-mask repair — each array task handles 1/NUM_JOBS of the
# entries that have missing or zero-byte depth/refined files (round-robin).
# Skips entries that already have both valid files — safe to re-run.
#
# JSON annotation files are NOT updated by the workers to avoid concurrent
# writes. Run refine_touch_masks.sh (without --overwrite) after all tasks
# finish to backfill the depth_path field.
#
# To change parallelism: update --array and NUM_JOBS together.

NUM_JOBS=4   # must match the upper bound of --array + 1

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate touch_from_segmentation

cd "$HOME/Touch_Breaks_Feelings"

EK_ROOT="/scratch-shared/$(whoami)/epic_kitchen"

python scripts/epic_kitchen/repair_depth_masks.py \
    --anno-dir "$EK_ROOT/annotations" \
    --splits train val \
    --dilation-radius 10 \
    --abs-d-threshold 0.05 \
    --local-radius 10 \
    --guided-filter \
    --refine-radius 4 \
    --refine-eps 0.1 \
    --num-jobs  "$NUM_JOBS" \
    --job-index "$SLURM_ARRAY_TASK_ID"
    # Add --dry-run to preview what would be reprocessed without writing.
