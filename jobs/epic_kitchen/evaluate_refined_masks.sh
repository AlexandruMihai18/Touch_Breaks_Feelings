#!/bin/bash

#SBATCH --partition=staging
#SBATCH --job-name=eval_ek_refined
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_output_%A.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=daniel.otero.gomez@student.uva.nl

# Evaluate depth-refined touch masks (EPIC Kitchen) as binary classification.
# No GPU required — reads refined PNGs and annotation JSONs only.
# Runtime: a few minutes for ~3k masks at full resolution.

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate touch_from_segmentation

cd "$HOME/Touch_Breaks_Feelings"

python scripts/epic_kitchen/evaluate_refined_masks.py \
    --splits train val \
    --min-samples 5 \
    --output results/ek_refined_eval.csv
