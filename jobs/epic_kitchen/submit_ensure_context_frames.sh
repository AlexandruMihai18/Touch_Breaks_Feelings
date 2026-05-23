#!/bin/bash
# Submit the parallel context-frame download + JSON regeneration pipeline.
#
# Usage (run on the login node, not via sbatch):
#   bash jobs/epic_kitchen/submit_ensure_context_frames.sh
#
# Dependency graph:
#
#   [ensure array 0-(N_TASKS-1)] ──afterok──▶ [generate_context_frames]
#
# Each array task downloads its slice of missing frames.  Once all tasks
# succeed, generate_context_frames.sh rebuilds val_context_frames.json.

set -euo pipefail

N_TASKS=8   # tune to the number of videos / desired parallelism

ENSURE_SCRIPT="jobs/epic_kitchen/ensure_context_frames.sh"
REGEN_SCRIPT="jobs/epic_kitchen/generate_context_frames.sh"

ARRAY_END=$(( N_TASKS - 1 ))

ENSURE_JOB=$(sbatch \
    --array=0-${ARRAY_END} \
    --export=N_TASKS=${N_TASKS} \
    "${ENSURE_SCRIPT}" | awk '{print $NF}')
echo "Ensure array job:   ${ENSURE_JOB}  (${N_TASKS} tasks)"

REGEN_JOB=$(sbatch \
    --dependency="afterok:${ENSURE_JOB}" \
    "${REGEN_SCRIPT}" | awk '{print $NF}')
echo "Regen JSON job:     ${REGEN_JOB}  (runs after all array tasks succeed)"

echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f slurm_output_${ENSURE_JOB}_*.out"
