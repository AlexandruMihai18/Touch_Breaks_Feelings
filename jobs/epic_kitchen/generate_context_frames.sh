#!/bin/bash

#SBATCH --partition=staging
#SBATCH --job-name=ek_ctx_frames
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=slurm_output_%A.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=daniel.otero.gomez@student.uva.nl

# Build val_context_frames.json for EPIC Kitchen.
# Scans VAL touch anchors and indexes ±4 context frames (0.5 s apart)
# from the pre-extracted frame JPEGs — no GPU or video decoding needed.
#
# Run after the download + annotation pipeline has produced val.json.

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate touch_from_segmentation

cd "$HOME/Touch_Breaks_Feelings"

EK_ROOT="/scratch-shared/$(whoami)/epic_kitchen"
ANNO_DIR="${EK_ROOT}/annotations"
FRAMES_DIR="${EK_ROOT}/frames"

python scripts/epic_kitchen/generate_context_frames.py \
    --annotations-dir "$ANNO_DIR" \
    --frames-dir      "$FRAMES_DIR"
