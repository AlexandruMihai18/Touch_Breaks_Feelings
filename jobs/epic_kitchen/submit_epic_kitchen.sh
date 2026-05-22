#!/bin/bash
# Submit the full parallel EPIC-Kitchen download + refinement pipeline.
#
# Usage (run on the login node, not via sbatch):
#   bash jobs/epic_kitchen/submit_epic_kitchen.sh
#
# Dependency graph:
#
#   [train array 0-7] ──┐
#                        ├─ afterok ─▶ [generate_annotations]
#   [val array 0-7]   ──┤
#                        └─ afterok ─▶ [refine_masks array 0-7]
#
# generate_annotations and refine_masks run in parallel once all downloads
# succeed.  If any download task fails both downstream jobs are cancelled.

set -euo pipefail

ARRAY_SCRIPT="jobs/epic_kitchen/download_epic_kitchen_array.sh"
ANNO_SCRIPT="jobs/epic_kitchen/generate_annotations.sh"
REFINE_SCRIPT="jobs/epic_kitchen/refine_masks_array.sh"

# Submit train array
TRAIN_JOB=$(sbatch --array=0-7 --export=SPLIT=train "${ARRAY_SCRIPT}" | awk '{print $NF}')
echo "Train array job:    ${TRAIN_JOB}"

# Submit val array
VAL_JOB=$(sbatch --array=0-7 --export=SPLIT=val "${ARRAY_SCRIPT}" | awk '{print $NF}')
echo "Val array job:      ${VAL_JOB}"

DEP="afterok:${TRAIN_JOB}:${VAL_JOB}"

# Annotation generation — depends on both download arrays
ANNO_JOB=$(sbatch --dependency="${DEP}" "${ANNO_SCRIPT}" | awk '{print $NF}')
echo "Annotation job:     ${ANNO_JOB}"

# Mask refinement array — depends on both download arrays, runs alongside annotation gen
REFINE_JOB=$(sbatch --array=0-7 --dependency="${DEP}" "${REFINE_SCRIPT}" | awk '{print $NF}')
echo "Refine masks job:   ${REFINE_JOB}"

echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f slurm_output_${TRAIN_JOB}_*.out"