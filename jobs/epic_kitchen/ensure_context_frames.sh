#!/bin/bash

#SBATCH --partition=staging
#SBATCH --job-name=ek_ensure_ctx
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_output_%A.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=daniel.otero.gomez@student.uva.nl

# Verify that every ±4 context frame for VAL touch anchors exists on disk.
# Downloads any missing frames from the VISOR frame ZIPs via HTTP range
# requests, then regenerates val_context_frames.json.
#
# Run after the download + annotation pipeline has produced val.json.
# Safe to re-run: already-present frames are skipped.

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate touch_from_segmentation

cd "$HOME/Touch_Breaks_Feelings"

EK_ROOT="/scratch-shared/$(whoami)/epic_kitchen"
ANNO_DIR="${EK_ROOT}/annotations"
FRAMES_DIR="${EK_ROOT}/frames"

python scripts/epic_kitchen/ensure_context_frames.py \
    --annotations-dir "$ANNO_DIR" \
    --frames-dir      "$FRAMES_DIR" \
    --workers         "${SLURM_CPUS_PER_TASK}"
