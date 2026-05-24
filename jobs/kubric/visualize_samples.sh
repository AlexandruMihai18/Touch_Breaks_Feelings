#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=kubric_viz_samples
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --output=slurm_output_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=daniel.otero.gomez@student.uva.nl

# Run SegGPT + annotate_touch on a small Kubric sample set and save visualizations.
#
# Output layout:
#   results/kubric_viz/sample_000_<class>/
#     context.jpg          — raw context frame
#     context_masks.jpg    — context with GT object masks overlaid
#     target.jpg           — raw target frame
#     target_inferred.jpg  — target with inferred object + touch masks overlaid

NUM_SAMPLES=10

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate touch_from_segmentation

cd "$HOME/Touch_Breaks_Feelings"

python scripts/kubric/visualize_samples.py \
    --data-root "/scratch-shared/$(whoami)" \
    --num-samples "$NUM_SAMPLES" \
    --output-dir results/kubric_viz \
    --seed 42 \
    --dilation 10 \
    --abs-d-threshold 0.05 \
    --local-radius 10
