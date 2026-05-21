#!/bin/bash

#SBATCH --partition=staging
#SBATCH --job-name=gh_prepare
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_output_%A.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=daniel.otero.gomez@student.uva.nl

# CPU-only preparation pipeline — no GPU needed.
#
# Step 1: extract annotated contact frames from each *_denoised.mp4 video.
#   ProcessPoolExecutor uses all 16 CPUs; ~500 videos finish in well under 1h.
# Step 2: generate GT annotation JSONs (train/val) from *_times.txt labels
#   in a single pass. Lightweight — takes a few minutes.
#
# Run after download.sh has completed.

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate touch_from_segmentation

cd "$HOME/Touch_Breaks_Feelings"

DATA_DIR="data/greatest_hits/vis-data-256/vis-data-256"
FRAMES_DIR="data/greatest_hits/frames"
ANNO_DIR="data/greatest_hits/annotations"

# Step 1 — extract frames
python scripts/greatest_hits/2_extract_greatest_hits_frames.py \
    --data_dir   "$DATA_DIR" \
    --output_dir "$FRAMES_DIR" \
    --num_workers "${SLURM_CPUS_PER_TASK}" \
    --skip_processed_videos

# Step 2 — GT annotation JSONs
python scripts/greatest_hits/3_generate_gh_gt_annotations.py \
    --data-dir    "$DATA_DIR" \
    --frames-dir  "$FRAMES_DIR" \
    --train-split "$DATA_DIR/train.txt" \
    --test-split  "$DATA_DIR/test.txt" \
    --output-dir  "$ANNO_DIR"
