# Land-Take-Prediction-Project-NINA-

## Project Overview
Predicting land-take (urban expansion, deforestation, etc.) from Sentinel-2 satellite imagery using deep learning. Uses time series data and binary land-take masks from the HABLOSS dataset.

## Baseline Models

### U-Net Early Fusion (`train_unet.py`)
- **Architecture**: U-Net with ResNet34 encoder (segmentation_models_pytorch)
- **Input**: Time series of first 7 Sentinel-2 timesteps (7 × 9 bands = 63 channels via early fusion)
- **Chip size**: 64×64 (center-cropped from variable-sized inputs)
- **Training**: 50 epochs, CrossEntropyLoss, Adam optimizer
- **Batch size**: 8
- **Augmentation**: Random horizontal/vertical flips + random 90° rotations
- **Normalization**: Scale by 10000 + per-channel standardization

### FCEF Temporal Model (`train_early_fusion.py`)
- **Architecture**: Fully Convolutional Early Fusion (FCEF) with temporal convolutions
- **Input**: Time series of first 7 Sentinel-2 timesteps (T=7, C=9 bands)
- **Chip size**: 64×64 (center-cropped from variable-sized inputs)
- **Training**: 50 epochs, CrossEntropyLoss, Adam optimizer
- **Batch size**: 4
- **Augmentation**: Random horizontal/vertical flips + random 90° rotations
- **Normalization**: Scale by 10000 + per-channel standardization

**Fair Comparison**: Both models use identical data splits (70/15/15), normalization, augmentation strategies, and random seeds (42) for reproducible comparison. Both use the first 7 timesteps from the Sentinel-2 time series.

## Training Scripts

Production training scripts for running on IDUN cluster with WandB logging:
- `train_unet.py`: U-Net early fusion training with automatic package installation
- `train_early_fusion.py`: FCEF temporal model training with automatic package installation
- `slurm_unet.sh`: SLURM job script for U-Net (4 hours, 1 GPU, 32GB RAM)
- `slurm_fcef.sh`: SLURM job script for FCEF (4 hours, 1 GPU, 32GB RAM)
- `IDUN_GUIDE.md`: Complete guide for running on IDUN

Both scripts include:
- Automatic installation of compatible PyTorch versions (2.1.0, torchvision 0.16.0)
- Robust validation logging with error handling
- Guaranteed test set visualization logging to WandB
- Center-cropping logic to handle variable-sized input chips

Run locally:
```bash
python train_unet.py
python train_early_fusion.py
```

Submit to IDUN:
```bash
sbatch slurm_unet.sh
sbatch slurm_fcef.sh
```

## Repository Structure

### `notebooks/`
- Exploratory notebooks and baseline experiments

### `src/`
- `config.py`: Data paths configuration (SENTINEL_DIR, MASK_DIR, VHR_DIR)
- `data/splits.py`: Shared train/val/test splitting (70/15/15, random_state=42)
- `data/transform.py`: Shared augmentation and normalization transforms for fair comparison
  - `NormalizeBy`: Scale by constant (10000 for Sentinel-2 TOA)
  - `Normalize`: Per-channel standardization using training set statistics
  - `RandomFlipTS`: Random horizontal/vertical flips (applied to all timesteps)
  - `RandomRotate90TS`: Random 90° rotations (applied to all timesteps)
  - `ComposeTS`: Transform composition for time series data
- `data/timeseries_dataset.py`: Unified time series dataset for both models
  - Handles variable-sized chips with center-cropping to 64×64
  - Reshapes Sentinel-2 data to (T, C, H, W) format
  - Supports temporal slicing (e.g., "first_half" for first 7 timesteps)
- `data/sentinel_habloss_dataset.py`: Legacy single-image dataset (deprecated)
- `models/external/torchrs_fc_cd.py`: FCEF model implementation

### `data/` (on IDUN cluster)
- `raw/Sentinel/`: Sentinel-2 time series GeoTIFFs (126 bands = 14 timesteps × 9 bands)
- `raw/masks/`: Binary land-take masks (upsampled to match Sentinel resolution)
- `raw/VHR/`: Very High Resolution imagery (6 bands = 2 timesteps × 3 RGB bands)
- `processed/`: Preprocessed data

### `logs/`
- SLURM job output files (`.out` and `.err` for each job)

## Data Format
- **Sentinel-2 chips**: Variable sizes (typically ~64×84 pixels), 126 bands
  - Automatically center-cropped to 64×64 during training
  - Layout: 7 years × 2 quarters/year × 9 bands = 126 bands total
  - Reshaped to (T=14, C=9, H, W) for temporal processing
  - "first_half" mode uses T=7 (first 7 timesteps)
- **Masks**: Binary land-take masks (0=background, 1=land-take)
- **Preprocessing**: All chips center-cropped to 64×64, with padding if smaller