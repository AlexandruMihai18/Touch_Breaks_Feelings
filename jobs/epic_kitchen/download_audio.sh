#!/bin/bash

#SBATCH --partition=staging
#SBATCH --job-name=ek_audio
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --output=slurm_output_%A.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=daniel.otero.gomez@student.uva.nl

# Standalone Phase 6 re-runner — stream-extracts AAC audio from EK100/EK55
# HTTP endpoints via ffmpeg.  No GPU needed; I/O-bound.
# Run after phases 1-3 (frames on disk).  Safe to re-run: skips existing .aac files.
#
# To target specific videos instead of the full frames/ tree:
#   sbatch download_audio.sh --video-ids P01_01 P01_103

module purge
module load 2023
module load FFmpeg/6.0-GCCcore-12.3.0
module load Anaconda3/2023.07-2

source activate touch_from_segmentation

cd "$HOME/Touch_Breaks_Feelings"

python scripts/epic_kitchen/download_audio.py \
    --data-dir "./data/epic_kitchen" \
    "$@"
