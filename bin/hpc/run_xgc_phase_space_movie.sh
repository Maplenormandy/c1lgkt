#!/bin/bash
#SBATCH --job-name=xgc_f3d_movie
#SBATCH --output=xgc_f3d_movie.out
#SBATCH --error=xgc_f3d_movie.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128   # Use 8 cores on this one task
#SBATCH --time=00:30:00       # Walltime
#SBATCH --partition=short     # Adjust based on your cluster's available partitions

# Load conda
module load conda

# Activate conda environment
conda activate research-forge

# Change to the directory where the script is located
cd /global/homes/n/normandy/c1lgkt/

# Run the script
python /bin/hpc/xgc_f3d_movie.py