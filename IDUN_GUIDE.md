# Running Training Scripts on IDUN

This guide explains how to run the training scripts on IDUN

## Prerequisites

1. Access to IDUN cluster
3. Data files in correct directories (see `src/config.py`)

## Setup on IDUN

### 1. Clone/Upload Project
From https://github.com/Ceciliamoller/Land-Take-Prediction-Project-NINA-

```bash
# SSH to IDUN
ssh username@idun.hpc.ntnu.no

# Navigate to your work directory
cd /cluster/home/$USER

# Clone or upload project
```

### 2. Create Virtual Environment
```bash
# Navigate into project
cd /cluster/home/$USER

# Load Python module
module load Python/3.10.8-GCCcore-12.2.0

# Create virtual environment
python -m venv .venv

# Activate it
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure WandB
```bash
# Use .env file
echo "WANDB_API_KEY=your_key_here" > .env
```

### 4. Verify Data Paths
Ensure your data paths in `src/config.py` match your IDUN setup:
```python
SENTINEL_DIR = Path("/cluster/home/your_user/data/raw/Sentinel")
MASK_DIR = Path("/cluster/home/your_user/data/raw/masks")
```

## Running Training Jobs

Optional: adjust job name and/or time allocation for the job at the top of slurm script:

```python
#!/bin/bash

#SBATCH --job-name=fcef
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
```

### Submit U-Net Training
```bash
sbatch slurm_unet.sh
```

### Submit FCEF Training
```bash
sbatch slurm_fcef.sh
```

### Check Job Status
```bash
# View all your jobs
squeue -u $USER

# View specific job
squeue -j <job_id>

# Cancel job
scancel <job_id>
```

### Monitor Training
```bash
# View output logs in real-time
tail -f logs/unet_<job_id>.out
tail -f logs/fcef_<job_id>.out

# View error logs
tail -f logs/unet_<job_id>.err
tail -f logs/fcef_<job_id>.err
```

