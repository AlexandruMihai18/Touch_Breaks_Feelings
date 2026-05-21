#!/bin/bash

#SBATCH --partition=staging
#SBATCH --job-name=gh_download
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_output_%A.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=daniel.otero.gomez@student.uva.nl

# Pure network I/O — wget/curl + unzip are single-threaded, 2 CPUs is enough.
# Budget 6h for the full dataset (~70 GB total).
# Change --low to --full to download full-resolution videos.

module purge
module load 2025
module load Anaconda3/2025.06-1

cd "$HOME/Touch_Breaks_Feelings"

bash scripts/greatest_hits/1_download_greatest_hits.sh --low
