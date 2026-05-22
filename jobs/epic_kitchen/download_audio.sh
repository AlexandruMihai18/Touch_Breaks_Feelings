#!/bin/bash

#SBATCH --partition=staging
#SBATCH --job-name=ek_audio
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --array=0-7                          # 8 parallel tasks (indices 0-7)
#SBATCH --output=slurm_output_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=daniel.otero.gomez@student.uva.nl

# Parallelised Phase 6 — each array task handles 1/NUM_JOBS of the video list
# (round-robin by SLURM_ARRAY_TASK_ID).  Safe to re-run: skips existing .m4a files.
# Each task writes its own failures_audio_<index>.json to avoid races.
#
# To change parallelism: update --array and NUM_JOBS together.
# To target specific videos instead of the full frames/ tree:
#   sbatch download_audio.sh --video-ids P01_01 P01_103

NUM_JOBS=8   # must match the upper bound of --array + 1

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate touch_from_segmentation

cd "$HOME/Touch_Breaks_Feelings"

python scripts/epic_kitchen/download_audio.py \
    --data-dir "./data/epic_kitchen" \
    --num-jobs  "$NUM_JOBS" \
    --job-index "$SLURM_ARRAY_TASK_ID" \
    "$@"
