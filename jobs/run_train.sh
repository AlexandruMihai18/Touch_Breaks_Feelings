#!/bin/bash

#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=seggpt_touch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=04:00:00
#SBATCH --output=slurm_output_%A.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=daniel.otero.gomez@student.uva.nl

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate touch_from_segmentation

cd "$HOME/FOMO/touch_from_segmentation"

python train.py \
    --datasets epic_kitchen \
    --max_steps 6000 \
    --batch_size 16 \
    --lr 1e-2 \
    --weight_decay 0.05 \
    --warmup_steps 100 \
    --num_workers 8 \
    --val_every_n_steps 50 \
    --output_dir "./checkpoints/run_${SLURM_JOB_ID}" \
    --group_tag "seggpt-touch-scale" \
    --filter_classes pan \
    --touch-variant refined_dilated_r40 \
    --accumulate_grad_batches 2 \
    --project_name "pan_only_experiments"
    # --object-filter-file "./data/epic_kitchen/object_labels_filtered.txt" \