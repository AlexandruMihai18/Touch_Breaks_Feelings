#!/bin/bash

#SBATCH --partition=staging
#SBATCH --job-name=gh_check_state
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=slurm_output_%A.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daniel.otero.gomez@student.uva.nl

# CPU-only state check — no GPU needed.
# Reports complete / partial / missing folders and frame coverage.
# Add --list partial  to print folders that failed mid-run.
# Add --list missing  to print folders that were never started.
# Add --list all      to print every folder with its status.

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate touch_from_segmentation

cd "$HOME/Touch_Breaks_Feelings"

GH_ROOT="/scratch-shared/$(whoami)/greatest_hits"

python scripts/greatest_hits/check_annotation_state.py \
    --frames-dir "${GH_ROOT}/frames" \
    --masks-dir  "${GH_ROOT}/masks"
    # Add --list partial / missing / complete / all for per-folder breakdown.
