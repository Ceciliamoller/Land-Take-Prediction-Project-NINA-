# Land-Take Prediction Using Temporal Satellite Imagery

## What This Project Does

The goal of this project is to predict land-take (urban expansion, deforestation, infrastructure development) from Sentinel-2 satellite time series. The models learn to segment binary land-take masks from the HABLOSS dataset, identifying where land conversion has occurred over time.

## Models

We compare two baseline architectures for temporal land-take prediction:

### U-Net Early Fusion (`train_unet.py`)
- Architecture: U-Net with ResNet34 encoder (from segmentation_models_pytorch)
- Input: 64×64 chips, 63 channels (7 timesteps concatenated)
- Training: 50 epochs, batch size 8, Adam optimizer, CrossEntropyLoss
- Data augmentation: Random flips and 90° rotations

### FCEF Temporal Model (`train_early_fusion.py`)
- Architecture: Fully Convolutional Early Fusion (FCEF)
- Input: 64×64 chips, T=7 timesteps, C=9 bands per timestep
- Training: 50 epochs, batch size 4, Adam optimizer, CrossEntropyLoss
- Data augmentation: Random flips and 90° rotations

### Fair Comparison Setup
Both models use the same data pipeline for reproducibility:
- Same 70/15/15 train/val/test splits (random seed = 42)
- Same normalization: scale by 10000, then standardize per channel
- Same augmentation strategy
- Same 7-timestep temporal window

This ensures we're comparing architectures, not implementation details.

## Dependencies

Main packages (see `requirements.txt` for complete list):
- PyTorch 2.1.0 + torchvision 0.16.0 (works with IDUN P100 GPUs)
- Python 3.10.8 (from IDUN module)
- segmentation-models-pytorch (U-Net)
- wandb (experiment tracking)
- rasterio (reads GeoTIFF files)
- opencv-python (upscales masks for visualization)

The SLURM scripts auto-install PyTorch and segmentation-models-pytorch to avoid version conflicts on IDUN.

## Running the Code

### Initialization
Follow `IDUN_GUIDE.md` to setup environment and data first time. Then do the following to run the code after setup:

```bash
# Load Python module
module load Python/3.10.8-GCCcore-12.2.0

#Activate virtual environment
source .venv/bin/activate
```

#### Run locally:
```bash
python train_unet.py
python train_early_fusion.py
```

#### Submit to IDUN cluster:
```bash
sbatch slurm_unet.sh
sbatch slurm_fcef.sh
```

### What's Included
- `train_unet.py` / `train_early_fusion.py`: Main training scripts
- `slurm_unet.sh` / `slurm_fcef.sh`: SLURM job scripts 
- `IDUN_GUIDE.md`: Full setup instructions for IDUN cluster

Both training scripts:
- Handle variable-sized chips with center-cropping
- Log metrics to WandB every epoch (loss, IoU, F1, precision, recall, accuracy)
- Save combined GT+prediction visualizations at the end of training (in that order)

## Code Organization

### `notebooks/`
Early experiments and exploratory analysis.

### `src/`
Main code library:

**Data pipeline** (`src/data/`):
- `timeseries_dataset.py` - Main dataset class. Loads Sentinel-2 time series and handles variable chip sizes with center-cropping.
- `splits.py` - Train/val/test split logic (70/15/15, fixed seed)
- `transform.py` - Augmentation and normalization transforms
  - `NormalizeBy` - Scales by 10000 (Sentinel-2 TOA reflectance)
  - `Normalize` - Per-channel standardization
  - `RandomFlipTS` / `RandomRotate90TS` - Spatial augmentation for time series

**Models** (`src/models/`):
- `external/torchrs_fc_cd.py` - FCEF architecture implementation

**Config** (`src/config.py`):
- Data paths for IDUN cluster

### Data on IDUN
All data are expected to be accessed from `/cluster/home/your_user/data/raw/`:
- `Sentinel/` - Sentinel-2 GeoTIFFs (126 bands = 14 timesteps × 9 bands each)
- `masks/` - Binary land-take masks
- `vhr/` - Very high resolution RGB imagery
- `PlanetScope/` - PlanetScope imagery
- `AlphaEarth/` - AlphaEarth embeddings at 10x10 m resolution

### `logs/`
SLURM output files from cluster jobs.

## Data Details

**Sentinel-2 chips**: Each chip covers the same location across 14 timesteps (7 years × 2 seasons per year). We use 9 spectral bands per timestep = 126 total bands per chip. Chips have variable sizes (typically ~64×84 pixels), so we center-crop to 64×64 during training.

For this project, we use only the first 7 timesteps (`temporal_mode = "first_half"`).

**Masks**: Binary segmentation masks. 1 = land-take occurred, 0 = background.

## Results

We trained both models on IDUN using the same data pipeline. Check the WandB project (`Baseline`) for full training curves and metrics.

**Key findings:**
- Both models converge within 50 epochs
- The temporal architecture (FCEF) explicitly models the time dimension, while U-Net treats it as concatenated channels
- Test set visualizations show how each model segments land-take regions
- All results are reproducible using the training scripts with seed=42

See the WandB dashboard for:
- Training and validation loss curves
- Per-epoch metrics (IoU, F1, Precision, Recall, Accuracy)
- Side-by-side ground truth and prediction visualizations

## Experiment Tracking

We use Weights & Biases (WandB) to track all experiments.

**Project**: `Baseline` (entity: `nina_prosjektoppgave`)

**What gets logged each epoch:**
- Training loss
- Validation loss
- Validation metrics: IoU, F1, Precision, Recall, Accuracy
- Test metrics (final epoch only)

**Visualization:**
At the end of training, we log test set predictions to WandB:
- Format: Side-by-side images (ground truth left, prediction right)
- Upscaled 4× (from 64×128 to 256×512) for better visibility
- Samples from up to 10 test batches
- Find them under the `test_combined_masks` key in WandB

We don't log RGB overlays or validation predictions during training - just the final test results.