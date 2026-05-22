#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=repair_depth
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=slurm_output_%A.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=daniel.otero.gomez@student.uva.nl

# Recompute depth PNGs and refined touch masks that are missing or zero-byte
# (e.g. from a prior run that hit the disk-full error).
# Skips entries that already have both valid files — safe to re-run.

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
    --refine-eps 0.1
    # Add --dry-run to preview what would be reprocessed without writing.
