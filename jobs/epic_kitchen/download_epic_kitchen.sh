#!/bin/bash

#SBATCH --partition=staging
#SBATCH --job-name=dl_epic
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=slurm_output_%A.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=daniel.otero.gomez@student.uva.nl

# No GPU needed — pipeline is HTTP I/O + CPU mask rendering (cv2) + ffmpeg audio extraction.
# Phase 6 streams audio from EK100 videos via ffmpeg (no full video stored).
# Time bumped to 8 h to accommodate audio extraction across all unique videos.
# --workers is matched to SLURM_CPUS_PER_TASK so we use every allocated core.

module purge
module load 2023
module load FFmpeg/6.0-GCCcore-12.3.0
module load Anaconda3/2023.07-2   # required for Phase 6 audio extraction

source activate touch_from_segmentation

cd "$HOME/Touch_Breaks_Feelings"

python -m scripts.epic_kitchen.download_epic_kitchen \
    --all-participants \
    --match-no-contact \
    --split train \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --output-dir "./data/epic_kitchen"

