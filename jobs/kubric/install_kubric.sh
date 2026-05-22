#!/bin/bash

#SBATCH --partition=rome
#SBATCH --job-name=install-kubric
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_output_%A.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sven.van.loon@student.uva.nl

module purge
module load 2025
module load Anaconda3/2025.06-1

cd "$HOME/Touch_Breaks_Feelings"

DATASET=movi_a/256x256 \
SPLIT=validation \
DATA_DIR=/scratch-shared/$USER/touch_breaks_feelings/kubric_tfds \
OUTPUT_ROOT=/scratch-shared/$USER/touch_breaks_feelings/kubric_movi_a_256 \
./run_kubric_movi_pipeline.sh