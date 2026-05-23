#!/bin/bash

#SBATCH --partition=staging
#SBATCH --job-name=ek_ensure_ctx
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_output_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=daniel.otero.gomez@student.uva.nl

# One task in an array job.  Do not submit directly — use submit_ensure_context_frames.sh.
#
# Required env var (set by submit script via --export):
#   N_TASKS  — total number of array tasks (must match --array=0-$(N_TASKS-1))
#
# Each task owns every N_TASKS-th video (round-robin by sorted video_id).
# Skips frames already on disk; downloads missing ones from VISOR ZIPs.
# JSON regeneration is handled by generate_context_frames.sh after all tasks complete.

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate touch_from_segmentation

cd "$HOME/Touch_Breaks_Feelings"

EK_ROOT="/scratch-shared/$(whoami)/epic_kitchen"

python scripts/epic_kitchen/ensure_context_frames.py \
    --annotations-dir "${EK_ROOT}/annotations" \
    --frames-dir      "${EK_ROOT}/frames" \
    --task-id         "${SLURM_ARRAY_TASK_ID}" \
    --n-tasks         "${N_TASKS}"
