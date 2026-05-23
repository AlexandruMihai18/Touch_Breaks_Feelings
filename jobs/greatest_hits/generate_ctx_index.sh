#!/bin/bash

#SBATCH --partition=staging
#SBATCH --job-name=gh_ctx_index
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=slurm_output_%A.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=daniel.otero.gomez@student.uva.nl

# Build train_ctx_index.json and val_ctx_index.json from train.json / val.json.
# Each file maps video_id → list of sample indices from that video.
#
# Run after finalize.sh (or after train.json / val.json exist in ANNO_DIR).

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate touch_from_segmentation

cd "$HOME/Touch_Breaks_Feelings"

ANNO_DIR="/scratch-shared/$(whoami)/greatest_hits/annotations"

python scripts/greatest_hits/generate_ctx_index.py \
    --annotations-dir "$ANNO_DIR"
