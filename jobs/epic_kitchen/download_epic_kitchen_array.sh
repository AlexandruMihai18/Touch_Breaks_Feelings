#!/bin/bash

#SBATCH --partition=staging
#SBATCH --job-name=dl_epic
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=8:00:00
#SBATCH --output=slurm_output_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=daniel.otero.gomez@student.uva.nl

# Array job: each task downloads one participant slice.
# Submit via submit_epic_kitchen.sh — do not run this script directly.
#
# Required env vars (set by submit_epic_kitchen.sh via --export):
#   SPLIT   — "train" or "val"
#
# Participant assignment: 28 participants split across N_JOBS=8 tasks (4 each,
# last task gets the remainder).  Each task writes frames/masks to the shared
# output dir; video IDs never collide across splits so parallel writes are safe.
# Failures are logged to a per-task file to avoid write conflicts.

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate touch_from_segmentation

cd "$HOME/Touch_Breaks_Feelings"

# ── participant assignment ───────────────────────────────────────────────────
ALL_PARTICIPANTS=(
    P01 P02 P03 P04
    P05 P06 P07 P08
    P10 P11 P12 P13
    P14 P15 P17 P18
    P20 P22 P23 P24
    P25 P26 P27 P28
    P30 P32 P35 P37
)
N_TOTAL=${#ALL_PARTICIPANTS[@]}   # 28
N_JOBS=8

TASK_ID=${SLURM_ARRAY_TASK_ID}
CHUNK=$(( (N_TOTAL + N_JOBS - 1) / N_JOBS ))   # ceiling div → 4
START=$(( TASK_ID * CHUNK ))
END=$(( START + CHUNK ))
(( END > N_TOTAL )) && END=$N_TOTAL

SLICE=("${ALL_PARTICIPANTS[@]:${START}:$(( END - START ))}")

echo "Task ${TASK_ID}: participants ${SLICE[*]} (${#SLICE[@]} total)"

# ── mode flags ───────────────────────────────────────────────────────────────
if [[ "${SPLIT}" == "train" ]]; then
    MODE_FLAG="--all"
elif [[ "${SPLIT}" == "val" ]]; then
    MODE_FLAG="--match-no-contact"
else
    echo "ERROR: SPLIT must be 'train' or 'val', got '${SPLIT}'" >&2
    exit 1
fi

# ── run ──────────────────────────────────────────────────────────────────────
python -m scripts.epic_kitchen.download_epic_kitchen \
    --participants "${SLICE[@]}" \
    ${MODE_FLAG} \
    --split "${SPLIT}" \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --output-dir "./data/epic_kitchen" \
    --failure-log "./data/epic_kitchen/failures_${SPLIT}_${TASK_ID}.json" \
    --skip-annotations \
    --skip-verify