#!/bin/bash

#SBATCH --partition=staging
#SBATCH --job-name=annotate_touch_centroid
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_output_annotate_touch_centroid_%A.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daniel.otero.gomez@student.uva.nl

# Populate x_touch_centroid and y_touch_centroid in all annotation JSONs for all
# three datasets (greatest_hits, epic_kitchen, kubric) across train + val splits.
#
# The centroid is the mask pixel closest to the mean of non-zero pixels, normalized
# to [0, 1] — the exact GT point used in DINO and Qwen training.
# No GPU needed. Safe to re-run: skips already-annotated entries unless --force.
#
# Check slurm_output_annotate_touch_centroid_<jobid>.out for per-split update counts.

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate touch_from_segmentation

cd "$HOME/Touch_Breaks_Feelings"

SCRATCH="/scratch-shared/$(whoami)"

python scripts/evaluation/run_annotate_touch_centroid.py \
    --data-root "$SCRATCH"
