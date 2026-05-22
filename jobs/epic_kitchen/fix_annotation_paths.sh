#!/bin/bash
#SBATCH --job-name=fix_annotation_paths
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=staging

OLD_PREFIX="/gpfs/home6/dotero/Touch_Breaks_Feelings/data"
NEW_PREFIX="/gpfs/scratch1/shared/dotero"

ANNOTATIONS_DIR="/scratch-shared/dotero/epic_kitchen/annotations"

echo "Fixing paths in: $ANNOTATIONS_DIR"
echo "  $OLD_PREFIX -> $NEW_PREFIX"

mapfile -t jsons < <(find "$ANNOTATIONS_DIR" -name "*.json")
echo "Found ${#jsons[@]} JSON files"

for f in "${jsons[@]}"; do
    sed -i "s|${OLD_PREFIX}|${NEW_PREFIX}|g" "$f"
    echo "Patched: $f"
done

echo "Done."
