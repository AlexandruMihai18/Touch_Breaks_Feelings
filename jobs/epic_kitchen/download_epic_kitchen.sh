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
# Runs two download passes writing frames/masks to the same output dir (no video ID
# collisions between VISOR splits), then generates annotations separately per split so
# train.json = VISOR train videos, val.json = VISOR val videos (no random re-splitting).
# --workers is matched to SLURM_CPUS_PER_TASK so we use every allocated core.

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate touch_from_segmentation

cd "$HOME/Touch_Breaks_Feelings"

DATA="./data/epic_kitchen"
VISOR_ANNO="${DATA}/visor/GroundTruth-SparseAnnotations/annotations"

# Pass 1: VISOR train — every contact + no-contact frame
python -m scripts.epic_kitchen.download_epic_kitchen \
    --all-participants \
    --all \
    --split train \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --output-dir "${DATA}" \
    --skip-annotations \
    --skip-verify

# Pass 2: VISOR val — all contact frames, no-contact matched to successful contact count
python -m scripts.epic_kitchen.download_epic_kitchen \
    --all-participants \
    --match-no-contact \
    --split val \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --output-dir "${DATA}" \
    --skip-annotations \
    --skip-verify

# Phase 5a: train.json — only VISOR train video IDs, no internal split
python -m scripts.epic_kitchen.download_epic_kitchen.generate_annotations \
    --frames_dir "${DATA}/frames" \
    --masks_dir  "${DATA}/masks" \
    --output_dir "${DATA}/annotations" \
    --video-ids-dir "${VISOR_ANNO}/train" \
    --output-name train

# Phase 5b: val.json — only VISOR val video IDs, no internal split
python -m scripts.epic_kitchen.download_epic_kitchen.generate_annotations \
    --frames_dir "${DATA}/frames" \
    --masks_dir  "${DATA}/masks" \
    --output_dir "${DATA}/annotations" \
    --video-ids-dir "${VISOR_ANNO}/val" \
    --output-name val

