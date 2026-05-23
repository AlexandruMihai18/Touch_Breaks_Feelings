#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=ek_seggpt_touch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --array=0-3                          # 4 parallel tasks (indices 0-3)
#SBATCH --output=slurm_output_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=daniel.otero.gomez@student.uva.nl

# Parallelised SegGPT inference — each array task handles 1/NUM_JOBS of the
# class list (round-robin by SLURM_ARRAY_TASK_ID, same seed so the split is
# deterministic and non-overlapping).
#
# Each task writes its own shard CSV:
#   results/epic_kitchen_val_seggpt_touch_results_<index>.csv
# Merge shards afterwards with, e.g.:
#   python scripts/merge_results.py results/epic_kitchen_val_seggpt_touch_results_*.csv
#
# To change parallelism: update --array and NUM_JOBS together.

NUM_JOBS=4   # must match the upper bound of --array + 1

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate touch_from_segmentation

cd "$HOME/Touch_Breaks_Feelings"

python scripts/seggpt_touch_inference.py epic_kitchen \
    --data-root "/scratch-shared/$(whoami)" \
    --seed 42 \
    --dilation 10 \
    --abs-d-threshold 0.05 \
    --local-radius 10 \
    --num-jobs  "$NUM_JOBS" \
    --job-index "$SLURM_ARRAY_TASK_ID"
