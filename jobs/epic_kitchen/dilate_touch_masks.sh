#!/bin/bash

#SBATCH --partition=staging
#SBATCH --job-name=dilate_touch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_output_%A.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=daniel.otero.gomez@student.uva.nl

# Pure CPU/I-O job — no GPU needed.
# Dilation is fast (cv2, parallel threads); 16 cores handle the full dataset in minutes.

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate touch_from_segmentation

cd "$HOME/Touch_Breaks_Feelings"

# ── Original touch masks ────────────────────────────────────────────────────
python scripts/epic_kitchen/dilate_touch_masks.py \
    --splits train val \
    --input-suffix touch \
    --radii 40 \
    --workers "${SLURM_CPUS_PER_TASK}"

# ── Depth-refined touch masks (run after refine_touch_masks.sh) ─────────────
python scripts/epic_kitchen/dilate_touch_masks.py \
    --splits train val \
    --input-suffix touch_refined \
    --radii 40 \
    --workers "${SLURM_CPUS_PER_TASK}"
