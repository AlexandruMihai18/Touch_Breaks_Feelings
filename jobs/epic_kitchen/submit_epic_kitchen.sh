#!/bin/bash
# Submit the full parallel EPIC-Kitchen download pipeline.
#
# Usage (run on the login node, not via sbatch):
#   bash jobs/epic_kitchen/submit_epic_kitchen.sh
#
# What gets submitted:
#   1. 8 array tasks for VISOR train  (--all)
#   2. 8 array tasks for VISOR val    (--match-no-contact)
#   3. 1 annotation-generation job    (runs after all 16 tasks succeed)
#
# The annotation job is held until every train AND val array task reports
# success (afterok).  If any task fails, the annotation job is cancelled
# and you will get a failure email.

set -euo pipefail

ARRAY_SCRIPT="jobs/epic_kitchen/download_epic_kitchen_array.sh"
ANNO_SCRIPT="jobs/epic_kitchen/generate_annotations.sh"

# Submit train array
TRAIN_JOB=$(sbatch --array=0-7 --export=SPLIT=train "${ARRAY_SCRIPT}" | awk '{print $NF}')
echo "Train array job: ${TRAIN_JOB}"

# Submit val array
VAL_JOB=$(sbatch --array=0-7 --export=SPLIT=val "${ARRAY_SCRIPT}" | awk '{print $NF}')
echo "Val array job:   ${VAL_JOB}"

# Submit annotation generation, held until both arrays fully succeed
ANNO_JOB=$(sbatch \
    --dependency=afterok:${TRAIN_JOB}:${VAL_JOB} \
    "${ANNO_SCRIPT}" | awk '{print $NF}')
echo "Annotation job:  ${ANNO_JOB}"

echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f slurm_output_${TRAIN_JOB}_*.out"