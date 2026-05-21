#!/bin/bash

#SBATCH --partition=staging
#SBATCH --job-name=dl_epic
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=slurm_output_%A.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=daniel.otero.gomez@student.uva.nl

# No GPU needed — pipeline is pure HTTP I/O + CPU mask rendering (cv2).
# --workers is matched to SLURM_CPUS_PER_TASK so we use every allocated core.

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate touch_from_segmentation

cd "$HOME/FOMO/touch_from_segmentation"

python -m scripts.epic_kitchen.download_epic_kitchen \
    --all-participants \
    --match-no-contact \
    --split train \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --output-dir "./data/epic_kitchen"

