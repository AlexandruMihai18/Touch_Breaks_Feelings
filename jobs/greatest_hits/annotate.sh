#!/bin/bash

#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=gh_annotate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_output_%A.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=daniel.otero.gomez@student.uva.nl

# Per-frame SAM + Depth-Anything-V2 annotation — GPU required.
# ~1-2s per frame; budget 8h for the full dataset.
# Re-submit with --skip-existing to resume an interrupted run.
#
# Run after prepare_data.sh has completed.

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate touch_from_segmentation

cd "$HOME/FOMO/touch_from_segmentation"

python scripts/greatest_hits/4_annotate_greatest_hits.py \
    --dilation 10 \
    --abs-d-threshold 0.05 \
    --local-radius 10 \
    --clahe-clip 2.0 \
    --guided-filter \
    --refine-radius 4 \
    --refine-eps 0.1 \
    --skip-existing
