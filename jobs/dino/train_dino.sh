#!/bin/bash

#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=dino-training
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_output_%A.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sven.van.loon@student.uva.nl

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate dino-touch-snellius

cd "$HOME/FOMO"

python -m touch_from_segmentation.dino.main train \
  --annotation-root touch_from_segmentation/data/greatest_hits/annotations \
  --hf-token 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 \
  --output-dir touch_from_segmentation/checkpoints/dino_greatest_mlp \
  --epochs 20 \
  --batch-size 32 \
  --head-type mlp \
  --wandb \
  --wandb-project dino-touch \
  --wandb-name greatest-mlp \
  --wandb-mode offline