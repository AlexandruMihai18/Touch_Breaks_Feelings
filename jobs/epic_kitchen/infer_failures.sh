#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=ek_infer_failures
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=slurm_output_%A.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=daniel.otero.gomez@student.uva.nl

# Run SegGPT inference on failure cases produced by sample_failures.py.
# Pass one or more failure-sampling output directories via DIRS (space-separated).
#
# Example:
#   DIRS="results/evaluation/my_run" sbatch jobs/epic_kitchen/infer_failures.sh
#
# To change dataset: set DATASET to epic_kitchen, greatest_hits, or kubric.

DATASET="${DATASET:-epic_kitchen}"
DATA_ROOT="/scratch-shared/$(whoami)"

if [[ -z "$DIRS" ]]; then
    echo "ERROR: DIRS is not set. Pass it as an environment variable, e.g.:"
    echo "  DIRS=\"results/evaluation/my_run\" sbatch $0"
    exit 1
fi

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate touch_from_segmentation

cd "$HOME/Touch_Breaks_Feelings"

# shellcheck disable=SC2086
python scripts/evaluation/failure_sampling/infer_failures.py \
    $DIRS \
    --dataset   "$DATASET" \
    --data-root "$DATA_ROOT" \
    --seed 42 \
    --dilation 10 \
    --abs-d-threshold 0.05 \
    --local-radius 10
