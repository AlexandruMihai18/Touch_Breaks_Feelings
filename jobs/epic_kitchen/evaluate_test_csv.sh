#!/bin/bash

#SBATCH --partition=staging
#SBATCH --job-name=eval_test_csv
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=slurm_output_eval_test_%A.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daniel.otero.gomez@student.uva.nl

# Smoke test for all evaluation scripts (depth, coverage, touch zones, object clusters).
# Prerequisite: run annotate_eval_fields.sh first to populate depth_touch,
# object_coverage, x_touch, y_touch in the val annotation JSON.
# No GPU required. Outputs PNGs to results/test/ next to the input CSV.
# Check slurm_output_eval_test_<jobid>.out and the PNGs in results/test/.

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate touch_from_segmentation

cd "$HOME/Touch_Breaks_Feelings"

CSV=results/test/epic_kitchen_val_seggpt_touch_results.csv
ANNOTATIONS=/scratch-shared/dotero/epic_kitchen/annotations/val.json
LABELS=data/epic_kitchen/object_labels.txt

# Build object cluster mappings from hand-crafted rule JSONs
python scripts/evaluation/cluster_object_classes.py \
    "$LABELS" \
    --output-dir data/epic_kitchen/

python scripts/evaluation/depth/analyze_depth_performance.py \
    "$CSV" \
    --annotations "$ANNOTATIONS" \
    --output results/test/eval_depth_perf.png

python scripts/evaluation/object_coverage/analyze_coverage_performance.py \
    "$CSV" \
    --annotations "$ANNOTATIONS" \
    --output results/test/eval_coverage_perf.png

python scripts/evaluation/touch_zones/analyze_grid_performance.py \
    "$CSV" \
    --annotations "$ANNOTATIONS" \
    --output results/test/eval_grid_heatmap.png
