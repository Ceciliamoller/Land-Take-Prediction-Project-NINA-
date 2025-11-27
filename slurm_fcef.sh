#!/bin/bash

#SBATCH --job-name=fcef_landtake
#SBATCH --account=ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/fcef_%j.out
#SBATCH --error=logs/fcef_%j.err

# Print job info
echo "=========================================="
echo "Starting FCEF training job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="
echo ""

# Load modules
module purge
module load Python/3.10.8-GCCcore-12.2.0

# Activate virtual environment
source .venv/bin/activate

# Set WandB API key (if not in .env)
# export WANDB_API_KEY="your_key_here"

# Create logs directory if it doesn't exist
mkdir -p logs

# Run training
python train_early_fusion.py

echo ""
echo "=========================================="
echo "Job finished"
echo "=========================================="
