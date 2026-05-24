#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=kubric_seggpt_touch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --array=0-3
#SBATCH --output=slurm_output_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=daniel.otero.gomez@student.uva.nl

# Parallelised SegGPT inference for Kubric MOVi-A 256.
# Each array task handles 1/NUM_JOBS of the context groups.
#
# Expected data layout:
#   /scratch-shared/$USER/kubric_movi_a_256/annotations/{train,val}.json
#   /scratch-shared/$USER/kubric_movi_a_256/annotations/{train,val}_ctx_index.json
#
# Each task writes its own shard CSV:
#   results/kubric_movi_a_256_val_seggpt_touch_results_<index>.csv
# Merge shards afterwards with:
#   python scripts/merge_results.py results/kubric_movi_a_256_val_seggpt_touch_results_*.csv

NUM_JOBS=4

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate touch_from_segmentation

cd "$HOME/Touch_Breaks_Feelings"

python scripts/seggpt_touch_inference.py kubric \
    --data-root "/scratch-shared/$(whoami)" \
    --seed 42 \
    --dilation 10 \
    --abs-d-threshold 0.05 \
    --local-radius 10 \
    --num-jobs "$NUM_JOBS" \
    --job-index "$SLURM_ARRAY_TASK_ID"
