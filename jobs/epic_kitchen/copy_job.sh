#!/bin/bash
#SBATCH --job-name=copy_epic_kitchen
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=staging

SRC=/home/dotero/Touch_Breaks_Feelings/data/epic_kitchen
DST=/scratch-shared/dotero/epic_kitchen

mkdir -p "$DST"

cp -r "$SRC/audio" "$DST/" &
# cp -r "$SRC/annotations" "$DST/" &
# cp -r "$SRC/frames"      "$DST/" &
# cp -r "$SRC/masks"       "$DST/" &

wait
echo "All done."