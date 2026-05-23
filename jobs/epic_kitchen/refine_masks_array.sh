#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --job-name=refine_masks
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-task=1
#SBATCH --time=8:00:00
#SBATCH --output=slurm_output_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=daniel.otero.gomez@student.uva.nl

# Array job: each task processes a round-robin shard of annotation entries
# that are missing a depth mask (_depth.png) or refined touch mask
# (_touch_refined.png).  Uses repair_depth_masks.py which skips entries that
# already have both valid files, so re-running is safe.
#
# Submitted by submit_epic_kitchen.sh after all download tasks succeed.
# Can also be run standalone to refine masks for any already-downloaded data:
#   sbatch --array=0-7 jobs/epic_kitchen/refine_masks_array.sh

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate touch_from_segmentation

cd "$HOME/Touch_Breaks_Feelings"

N_JOBS=8
EK_ROOT="/scratch-shared/$(whoami)/epic_kitchen"

python scripts/epic_kitchen/repair_depth_masks.py \
    --anno-dir  "${EK_ROOT}/annotations" \
    --splits    train val \
    --num-jobs  "${N_JOBS}" \
    --job-index "${SLURM_ARRAY_TASK_ID}"