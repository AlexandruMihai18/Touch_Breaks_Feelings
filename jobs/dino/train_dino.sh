#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=dino-training
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_output_%A.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sven.van.loon@student.uva.nl

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate dino-touch-snellius

cd "$HOME/Touch_Breaks_Feelings"

python -m dino.main train \
  --annotation-root /scratch-shared/$USER/epic_kitchen/annotations \
  --hf-token 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 \
  --output-dir checkpoints/dino_epic_point_linear \
  --epochs 12 \
  --batch-size 32 \
  --head-type linear \
  --task point \
  --wandb \
  --wandb-project dino-touch \
  --wandb-name epic-point-linear \
  --wandb-mode offline \
  --patience 3