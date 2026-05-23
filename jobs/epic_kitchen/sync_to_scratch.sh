#!/bin/bash

#SBATCH --partition=staging
#SBATCH --job-name=sync_epic
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=2:00:00
#SBATCH --output=slurm_output_%A.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=daniel.otero.gomez@student.uva.nl

# Sync new data (36 videos not yet in scratch) from home to scratch.
# --ignore-existing ensures files already present in scratch are never
# overwritten — only genuinely new files are transferred.
# Annotations are NOT synced; regenerate them after this job completes.

LOCAL="/gpfs/home6/dotero/Touch_Breaks_Feelings/data/epic_kitchen"
SCRATCH="/gpfs/scratch1/shared/dotero/epic_kitchen"

echo "LOCAL  : ${LOCAL}"
echo "SCRATCH: ${SCRATCH}"
echo ""

for subdir in frames masks audio; do
    echo "── Syncing ${subdir}/ ────────────────────────────────────"
    rsync -av --progress --ignore-existing \
        "${LOCAL}/${subdir}/" \
        "${SCRATCH}/${subdir}/"
    echo ""
done

echo "Done. Run generate_annotations next to update train.json / val.json."