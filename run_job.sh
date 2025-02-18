#!/bin/bash
#SBATCH --job-name=safety_job
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# Directly activate the Conda environment
conda activate safety_env

# Run the Python script
python main.py

