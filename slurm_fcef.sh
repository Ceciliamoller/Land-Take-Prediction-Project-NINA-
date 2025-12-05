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

echo "=========================================="
echo "Starting FCEF Early Fusion training job"
echo "Job ID:        $SLURM_JOB_ID"
echo "Job name:      $SLURM_JOB_NAME"
echo "Node(s):       $SLURM_NODELIST"
echo "Partition:     $SLURM_JOB_PARTITION"
echo "GPUs:          $SLURM_GPUS"
echo "=========================================="
echo ""


module purge
module load Python/3.10.8-GCCcore-12.2.0

WORKDIR=${SLURM_SUBMIT_DIR}
cd "$WORKDIR"

# Activate project venv
source .venv/bin/activate

echo "Running from directory: $WORKDIR"
echo ""

echo "GPU status:"
nvidia-smi || echo "nvidia-smi not available"
echo ""

mkdir -p logs

echo "Starting python train_early_fusion.py"
python train_early_fusion.py

echo ""
echo "=========================================="
echo "Job finished"
echo "=========================================="
