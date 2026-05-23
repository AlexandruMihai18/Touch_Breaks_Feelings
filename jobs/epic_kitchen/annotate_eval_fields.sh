#!/bin/bash

#SBATCH --partition=staging
#SBATCH --job-name=annotate_eval_fields
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_output_annotate_eval_%A.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daniel.otero.gomez@student.uva.nl

# Populate depth_touch, object_coverage, x_touch, y_touch fields into the val
# annotation JSON. Run this once before evaluate_test_csv.sh. No GPU needed.
# Check slurm_output_annotate_eval_<jobid>.out for per-field update counts.

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate touch_from_segmentation

cd "$HOME/Touch_Breaks_Feelings"

ANNOTATIONS="/scratch-shared/$(whoami)/epic_kitchen/annotations/val.json"

# Object coverage — uses object_mask_path already recorded in the annotation.
python scripts/evaluation/object_coverage/annotate_object_coverage.py \
    "$ANNOTATIONS"

# Touch zones — finds largest touch blob centroid, assigns 8×8 grid cell.
python scripts/evaluation/touch_zones/annotate_touch_zones.py \
    "$ANNOTATIONS" \
    --grid 8

# Depth under touch mask — requires colorized inferno depth PNGs at
# {frames_dir}/{image_stem}_depth.png. Entries without a matching file are
# silently skipped (depth_touch=null).
python scripts/evaluation/depth/annotate_touch_depth.py \
    "$ANNOTATIONS"
