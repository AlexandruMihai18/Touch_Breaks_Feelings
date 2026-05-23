#!/bin/bash

#SBATCH --partition=staging
#SBATCH --job-name=run_all_annotations
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_output_run_all_annotations_%A.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daniel.otero.gomez@student.uva.nl

# Populate depth_touch, x_touch, y_touch, and object_coverage fields for all
# datasets (greatest_hits, epic_kitchen, manual_annotations) and both splits
# (train + val). No GPU needed. Safe to re-run: skips already-annotated entries
# unless --force is passed.
#
# Check slurm_output_run_all_annotations_<jobid>.out for per-field update counts.

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate touch_from_segmentation

cd "$HOME/Touch_Breaks_Feelings"

SCRATCH="/scratch-shared/$(whoami)"

python scripts/evaluation/run_all_annotations.py \
    --data-root "$SCRATCH" \
    --datasets gh ek
