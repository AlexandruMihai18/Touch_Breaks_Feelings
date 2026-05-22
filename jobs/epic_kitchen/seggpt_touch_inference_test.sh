#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=ek_seggpt_touch_test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --output=slurm_output_test_%A.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daniel.otero.gomez@student.uva.nl

# Smoke test — runs inference on 3 classes only to verify the full pipeline
# (GPU, SegGPT, annotate_touch, CSV write) works before submitting the full array job.
# Check slurm_output_test_<jobid>.out and results/epic_kitchen_val_seggpt_touch_results_test.csv.

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate touch_from_segmentation

cd "$HOME/Touch_Breaks_Feelings"

python scripts/seggpt_touch_inference.py epic_kitchen \
    --split val \
    --seed 42 \
    --dilation 10 \
    --abs-d-threshold 0.05 \
    --local-radius 10 \
    --max-classes 3 \
    --output-dir results/test
