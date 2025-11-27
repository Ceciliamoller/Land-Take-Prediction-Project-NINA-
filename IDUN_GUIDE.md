# Running Training Scripts on IDUN

This guide explains how to run the training scripts on IDUN

## Prerequisites

1. Access to IDUN cluster
2. Project files uploaded to IDUN
3. Data files in correct directories (see `src/config.py`)
4. Virtual environment set up with dependencies

## Setup on IDUN

### 1. Clone/Upload Project
```bash
# SSH to IDUN
ssh username@idun.hpc.ntnu.no

# Navigate to your work directory
cd /cluster/work/users/$USER

# Clone or upload project
# (adjust paths as needed)
```

### 2. Create Virtual Environment
```bash
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
# Option 1: Set environment variable in SLURM script
# Add to slurm script: export WANDB_API_KEY="your_key_here"

# Option 2: Use .env file
echo "WANDB_API_KEY=your_key_here" > .env
```

### 4. Verify Data Paths
Edit `src/config.py` to ensure data paths match your IDUN setup:
```python
SENTINEL_DIR = Path("/cluster/work/users/youruser/data/raw/Sentinel")
MASK_DIR = Path("/cluster/work/users/youruser/data/raw/masks")
```

## Running Training Jobs

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

## Training Scripts Overview

### `train_unet.py`
- U-Net with ResNet34 encoder
- Single-image Sentinel-2 input (12 bands)
- 64×64 patches, 10 epochs
- Configuration at top of file

### `train_early_fusion.py`
- FCEF model for time series
- 7 Sentinel-2 timesteps (12 bands each)
- 64×64 patches, 10 epochs
- Configuration at top of file

```

## Results

- **WandB Dashboard**: View metrics, loss curves, and predictions online
- **Log Files**: Check `logs/` directory for training outputs
- **Model Checkpoints**: (Optional) Add model saving in training scripts

## Running Locally (for testing)

Before submitting to IDUN, test locally:
```bash
# Activate environment
source .venv/bin/activate

# Run training (will use CPU if no GPU)
python train_unet.py
python train_early_fusion.py
```

