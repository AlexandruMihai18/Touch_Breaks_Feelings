#!/bin/bash

#SBATCH --partition=staging
#SBATCH --job-name=gh_finalize
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_output_%A.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=daniel.otero.gomez@student.uva.nl

# CPU-only post-annotation pipeline — no GPU needed.
#
# Step 1: enrich train/val annotation JSONs with mask paths resolved from the
#   masks directory, and build ctx_index files for training-time context sampling.
# Step 2: evaluate the annotation pipeline as binary classification and write
#   per-split and per-material results to results/gh_eval.csv.
#
# Run after annotate.sh has completed.

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate touch_from_segmentation

cd "$HOME/Touch_Breaks_Feelings"

GH_ROOT="/scratch-local/$(whoami)/greatest_hits"
MASKS_DIR="${GH_ROOT}/masks"
ANNO_DIR="${GH_ROOT}/annotations"
DATA_DIR="${GH_ROOT}/vis-data-256/vis-data-256"

# Step 1 — enrich annotations with mask paths + build ctx_index
python scripts/greatest_hits/5_generate_gh_mask_annotations.py \
    --masks-dir       "$MASKS_DIR" \
    --annotations-dir "$ANNO_DIR"

# Step 2 — evaluate annotation pipeline
python scripts/greatest_hits/6_evaluate_annotation_pipeline.py \
    --masks-dir        "$MASKS_DIR" \
    --annotations-dir  "$ANNO_DIR" \
    --train-split      "$DATA_DIR/train.txt" \
    --test-split       "$DATA_DIR/test.txt" \
    --output           results/gh_eval.csv
