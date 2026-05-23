#!/bin/bash

#SBATCH --partition=staging
#SBATCH --job-name=gh_ctx_frames
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_output_%A.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=daniel.otero.gomez@student.uva.nl

# Extract temporal context frames (±4 steps × 0.5 s) around every annotated
# touch event and write train_context_frames.json / val_context_frames.json.
#
# Run after prepare_data.sh (annotated frames and *_times.txt must exist).

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate touch_from_segmentation

cd "$HOME/Touch_Breaks_Feelings"

GH_ROOT="/scratch-shared/$(whoami)/greatest_hits"
DATA_DIR="${GH_ROOT}/vis-data-256/vis-data-256"
FRAMES_DIR="${GH_ROOT}/frames"
ANNO_DIR="${GH_ROOT}/annotations"

python scripts/greatest_hits/2_1_generate_context_frames.py \
    --data-dir    "$DATA_DIR" \
    --frames-dir  "$FRAMES_DIR" \
    --output-dir  "$ANNO_DIR" \
    --train-split "$DATA_DIR/train.txt" \
    --test-split  "$DATA_DIR/test.txt"
