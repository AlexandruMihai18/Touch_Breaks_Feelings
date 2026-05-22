#!/bin/bash

#SBATCH --partition=staging
#SBATCH --job-name=gen_epic_anno
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=slurm_output_%A.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=daniel.otero.gomez@student.uva.nl

# Runs after all download array tasks finish (submitted with --dependency by
# submit_epic_kitchen.sh).  Generates train.json and val.json separately,
# filtered to their respective VISOR split video IDs — no random re-splitting.

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate touch_from_segmentation

cd "$HOME/Touch_Breaks_Feelings"

DATA="./data/epic_kitchen"
VISOR_ANNO="${DATA}/visor/GroundTruth-SparseAnnotations/annotations"

# train.json — VISOR train video IDs only
python -m scripts.epic_kitchen.download_epic_kitchen.generate_annotations \
    --frames_dir "${DATA}/frames" \
    --masks_dir  "${DATA}/masks" \
    --output_dir "${DATA}/annotations" \
    --video-ids-dir "${VISOR_ANNO}/train" \
    --output-name train

# val.json — VISOR val video IDs only
python -m scripts.epic_kitchen.download_epic_kitchen.generate_annotations \
    --frames_dir "${DATA}/frames" \
    --masks_dir  "${DATA}/masks" \
    --output_dir "${DATA}/annotations" \
    --video-ids-dir "${VISOR_ANNO}/val" \
    --output-name val