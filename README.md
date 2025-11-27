# Land-Take-Prediction-Project-NINA-

## Project Overview
Predicting land-take (urban expansion, deforestation, etc.) from Sentinel-2 satellite imagery using deep learning. Uses time series data and binary land-take masks from the HABLOSS dataset.

## Baseline Models

### U-Net Baseline (`03_smp_unet_baseline.ipynb`)
- **Architecture**: U-Net with ResNet34 encoder (segmentation_models_pytorch)
- **Input**: Single Sentinel-2 image (12 bands)
- **Patch size**: 64×64
- **Training**: 10 epochs, CrossEntropyLoss, Adam optimizer
- **Normalization**: Scale by 10000 + per-channel standardization

### FCEF Baseline (`fc_early_fusion.ipynb`)
- **Architecture**: Fully Convolutional Early Fusion (FCEF)
- **Input**: Time series of 7 Sentinel-2 images (12 bands each)
- **Patch size**: 64×64
- **Training**: 10 epochs, CrossEntropyLoss, Adam optimizer
- **Normalization**: Scale by 10000 + per-channel standardization

**Fair Comparison**: Both models use identical data splits (70/15/15), normalization, patch size, and random seeds for reproducible comparison.

## Training Scripts

Standalone training scripts for running on IDUN cluster:
- `train_unet.py`: U-Net training script
- `train_early_fusion.py`: FCEF training script
- `slurm_unet.sh`: SLURM job script for U-Net
- `slurm_fcef.sh`: SLURM job script for FCEF
- `IDUN_GUIDE.md`: Complete guide for running on IDUN

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
- `config.py`: Data paths configuration
- `data/splits.py`: Shared train/val/test splitting (70/15/15, random_state=42)
- `data/transform.py`: Shared normalization pipeline
- `data/sentinel_habloss_dataset.py`: Single-image Sentinel-2 dataset for U-Net
- `data/timeseries_dataset.py`: Time series Sentinel-2 dataset for FCEF

### `data/`
- `raw/Sentinel/`: Sentinel-2 time series GeoTIFFs
- `raw/masks/`: Binary land-take masks
- `processed/`: Preprocessed data