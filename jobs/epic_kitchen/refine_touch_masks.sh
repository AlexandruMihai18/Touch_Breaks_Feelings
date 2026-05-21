#!/bin/bash

#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=refine_touch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=slurm_output_%A.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=daniel.otero.gomez@student.uva.nl

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate touch_from_segmentation

cd "$HOME/FOMO/touch_from_segmentation"

python scripts/epic_kitchen/refine_touch_masks.py \
    --splits train val \
    --dilation-radius 10 \
    --abs-d-threshold 0.05 \
    --local-radius 10 \
    --guided-filter \
    --refine-radius 4 \
    --refine-eps 0.1
    # Add --overwrite to regenerate already-existing refined masks.
    # Add --dry-run to see what would be written without writing anything.
