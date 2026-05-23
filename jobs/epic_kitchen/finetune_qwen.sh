#!/bin/bash


#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=ft_qwen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=128G
#SBATCH --output=slurm_output_%A.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=daniel.otero.gomez@student.uva.nl

# Evaluate depth-refined touch masks (EPIC Kitchen) as binary classification.
# No GPU required — reads refined PNGs and annotation JSONs only.
# Runtime: a few minutes for ~3k masks at full resolution.

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate touch_from_segmentation

cd "$HOME/Touch_Breaks_Feelings"
export HF_HOME="/scratch-shared/$USER/huggingface_cache"
export PYTHONUTF8=1

python qwen_baseline/finetuning_qwen.py \
    --train_path /gpfs/scratch1/shared/dotero/epic_kitchen/annotations/train.json \
    --eval_path /gpfs/scratch1/shared/dotero/epic_kitchen/annotations/val.json \
    --wandb \
    --wandb_project "qwen-ft-epic-kitchen-touch"
