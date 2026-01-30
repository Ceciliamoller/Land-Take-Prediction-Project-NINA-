"""
U-Net Training Script for Land-Take Prediction

Based on 03_smp_unet_baseline.ipynb
Fair comparison setup with FCEF baseline: shared splits, normalization, patch size, random seeds
Uses TimeSeriesDataset with early fusion (T*C channels) for temporal prediction
"""

import sys
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm
import wandb

print(">>> train_unet.py started")


# Disable cuDNN completely for P100 compatibility
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

# Add project root to path
root = Path(__file__).resolve().parent
sys.path.append(str(root))

from src.config import SENTINEL_DIR, MASK_DIR
from src.data.timeseries_dataset import TimeSeriesDataset
from src.data.splits import get_splits, get_ref_ids_from_directory
from src.data.transform import (
    compute_normalization_stats,
    ComposeTS,
    NormalizeBy,
    RandomCropTS,
    CenterCropTS,
    Normalize,
    RandomFlipTS,
    RandomRotate90TS
)

import wandb

def upscale_mask(mask, scale: int = 4):
    """
    Upscale a 2D numpy mask (H, W) with values 0 or 255
    to a larger size using nearest-neighbor interpolation.
    
    Args:
        mask: 2D numpy array (H, W)
        scale: Upscaling factor (default 4)
    
    Returns:
        Upscaled mask (H*scale, W*scale)
    """
    t = torch.from_numpy(mask)[None, None].float()  # (1,1,H,W)
    t_up = F.interpolate(t, scale_factor=scale, mode="nearest")
    return t_up[0, 0].byte().numpy()


def log_masks(model, loader, device, step, name_prefix="val", max_batches=10):
    """
    Log ground-truth and predicted segmentation masks to WandB as combined side-by-side images.
    Iterates over multiple batches and creates a single image per sample with GT on left, prediction on right.
    Visualizes masks as black (0) and white (255).
    
    Args:
        model: The model to evaluate
        loader: DataLoader to sample from
        device: Device for inference
        step: WandB step (typically epoch number)
        name_prefix: Prefix for WandB keys (e.g., "val", "test")
        max_batches: Maximum number of batches to process (default 10)
    """
    try:
        import numpy as np
        
        model.eval()
        combined_images = []
        
        with torch.no_grad():
            loader_iter = iter(loader)
            for b_idx in range(max_batches):
                try:
                    imgs, masks = next(loader_iter)
                except StopIteration:
                    print(f"[INFO] log_masks ({name_prefix}): reached end of loader at batch {b_idx}")
                    break
                except RuntimeError as e:
                    print(f"[WARN] log_masks ({name_prefix}) batch {b_idx} failed: {e}")
                    continue
                
                if imgs.shape[0] == 0:
                    continue
                
                B, T, C, H, W = imgs.shape
                # For U-Net early fusion, flatten T and C
                x_unet = imgs.reshape(B, T * C, H, W).to(device)
                logits = model(x_unet)
                preds = logits.argmax(dim=1).cpu()  # (B, H, W)
                masks = masks.cpu()
                
                # Convert to uint8 and scale to 0/255 for visibility
                masks_vis = (masks * 255).byte().numpy()  # (B, H, W)
                preds_vis = (preds * 255).byte().numpy()  # (B, H, W)
                
                for i in range(B):
                    if len(masks_vis[i].shape) != 2 or len(preds_vis[i].shape) != 2:
                        continue
                    
                    # Combine GT (left) and prediction (right) side-by-side
                    combined = np.concatenate([masks_vis[i], preds_vis[i]], axis=1)  # (64, 128)
                    # Upscale for better visualization
                    upscaled = upscale_mask(combined, scale=4)  # (256, 512)
                    combined_images.append(wandb.Image(upscaled, caption=f"{name_prefix}_GT_left_PRED_right_b{b_idx}_i{i}"))
        
        # Log combined GT+prediction images
        if len(combined_images) > 0:
            wandb.log({f"{name_prefix}_combined_masks": combined_images}, step=step)
            print(f"[INFO] Logged {len(combined_images)} combined mask images for {name_prefix}")
        else:
            print(f"[WARN] log_masks ({name_prefix}): no valid samples to log")
    
    except Exception as e:
        print(f"[ERROR] log_masks ({name_prefix}) failed: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Random seed
    "random_seed": 42,
    
    # Data splits
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    
    # Model
    "architecture": "UNet_EarlyFusion",
    "encoder_name": "resnet34",
    "encoder_weights": "imagenet",
    "num_classes": 2,
    
    # Data
    "sensor": "sentinel",
    "temporal_mode": "first_half",  # 7 timesteps
    "chip_size": 64,  # Pre-cropped chips are 64×64
    
    # Training
    "epochs": 50,
    "learning_rate": 1e-3,
    "batch_size": 8,
    "augment_train": True,
    
    # Normalization
    "normalization": "scale_10000_plus_standardize",
    "num_samples_for_stats": 2000,
    
    # DataLoader
    "num_workers": 4,
    
    # WandB
    "wandb_project": "Baseline",
    "wandb_entity": "nina_prosjektoppgave",
}


# ============================================================================
# SETUP
# ============================================================================

def set_random_seeds(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"All random seeds set to {seed}")


def get_device():
    """Get device for training"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


# ============================================================================
# METRICS
# ============================================================================

def compute_confusion_binary(y_pred, y_true, positive_class=1):
    """
    Compute confusion matrix for binary classification.
    y_pred, y_true: (B, H, W) with 0/1 labels
    returns TP, FP, TN, FN as scalars
    """
    y_pred = (y_pred == positive_class)
    y_true = (y_true == positive_class)

    tp = (y_pred & y_true).sum().item()
    fp = (y_pred & ~y_true).sum().item()
    tn = (~y_pred & ~y_true).sum().item()
    fn = (~y_pred & y_true).sum().item()
    return tp, fp, tn, fn


def compute_metrics_from_confusion(tp, fp, tn, fn, eps=1e-8):
    """
    Compute metrics from confusion matrix values.
    Returns: dict with accuracy, precision, recall, f1, iou
    """
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


# ============================================================================
# TRAINING & VALIDATION
# ============================================================================

def train_one_epoch(model, loader, loss_fn, optimizer, device, scaler):
    """Train for one epoch with time series input"""
    model.train()
    total_loss = 0.0

    for x, masks in loader:
        # x shape: (B, T, C, H, W)
        # Reshape to (B, T*C, H, W) for U-Net early fusion
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)
        
        x = x.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            logits = model(x)
            loss = loss_fn(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    return avg_loss


def validate(model, loader, loss_fn, device):
    """Validate model with time series input"""
    model.eval()
    total_loss = 0.0
    sum_tp = sum_fp = sum_tn = sum_fn = 0

    with torch.no_grad():
        for x, masks in loader:
            # x shape: (B, T, C, H, W)
            # Reshape to (B, T*C, H, W) for U-Net early fusion
            B, T, C, H, W = x.shape
            x = x.reshape(B, T * C, H, W)
            
            x = x.to(device)
            masks = masks.to(device)

            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                logits = model(x)
                loss = loss_fn(logits, masks)
            
            total_loss += loss.item()

            pred = torch.argmax(logits, dim=1)
            tp, fp, tn, fn = compute_confusion_binary(pred, masks, positive_class=1)
            sum_tp += tp
            sum_fp += fp
            sum_tn += tn
            sum_fn += fn

    avg_loss = total_loss / len(loader)
    metrics = compute_metrics_from_confusion(sum_tp, sum_fp, sum_tn, sum_fn)
    
    return avg_loss, metrics


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Set random seeds
    set_random_seeds(CONFIG["random_seed"])
    
    # Get device
    device = get_device()
    
    # Get data splits
    print("\n" + "="*80)
    print("DATA SPLITS")
    print("="*80)
    all_ref_ids = get_ref_ids_from_directory(SENTINEL_DIR)
    print(f"Total reference IDs found: {len(all_ref_ids)}")
    
    train_ref_ids, val_ref_ids, test_ref_ids = get_splits(
        all_ref_ids,
        train_ratio=CONFIG["train_ratio"],
        val_ratio=CONFIG["val_ratio"],
        test_ratio=CONFIG["test_ratio"],
        random_state=CONFIG["random_seed"],
    )
    
    print(f"Train tiles: {len(train_ref_ids)} (~{100*len(train_ref_ids)/len(all_ref_ids):.0f}%)")
    print(f"Val tiles: {len(val_ref_ids)} (~{100*len(val_ref_ids)/len(all_ref_ids):.0f}%)")
    print(f"Test tiles: {len(test_ref_ids)} (~{100*len(test_ref_ids)/len(all_ref_ids):.0f}%)")
    print(f"✓ Using SHARED splits with FCEF baseline (random_state={CONFIG['random_seed']})")
    
    # Compute normalization stats
    print("\n" + "="*80)
    print("NORMALIZATION")
    print("="*80)
    temp_train_transform = ComposeTS([
        CenterCropTS(CONFIG["chip_size"]),
        NormalizeBy(10000.0),
    ])
    
    temp_train_ds = TimeSeriesDataset(
        train_ref_ids,
        sensor=CONFIG["sensor"],
        slice_mode=CONFIG["temporal_mode"],
        transform=temp_train_transform,
    )
    
    print("Estimating per-channel mean and std from training data...")
    mean, std = compute_normalization_stats(temp_train_ds, num_samples=CONFIG["num_samples_for_stats"])
    print(f"✓ Computed normalization stats: {len(mean)} channels")
    print(f"  Mean (first 5): {[f'{m:.4f}' for m in mean[:5]]}") 
    print(f"  Std (first 5): {[f'{s:.4f}' for s in std[:5]]}")
    
    # Create datasets
    print("\n" + "="*80)
    print("DATASETS")
    print("="*80)
    
    # Training transform with spatial augmentation (flips + rotations)
    if CONFIG["augment_train"]:
        train_transform = ComposeTS([
            CenterCropTS(CONFIG["chip_size"]),  # Pad/crop to 64×64
            RandomFlipTS(p_horizontal=0.5, p_vertical=0.5),
            RandomRotate90TS(),
            NormalizeBy(10000.0),
            Normalize(mean, std),
        ])
    else:
        train_transform = ComposeTS([
            CenterCropTS(CONFIG["chip_size"]),  # Pad/crop to 64×64
            NormalizeBy(10000.0),
            Normalize(mean, std),
        ])
    
    # Val/test transforms: no augmentation, only normalization
    val_transform = ComposeTS([
        CenterCropTS(CONFIG["chip_size"]),  # Pad/crop to 64×64
        NormalizeBy(10000.0),
        Normalize(mean, std),
    ])
    
    test_transform = ComposeTS([
        CenterCropTS(CONFIG["chip_size"]),  # Pad/crop to 64×64
        NormalizeBy(10000.0),
        Normalize(mean, std),
    ])
    
    train_ds = TimeSeriesDataset(
        train_ref_ids,
        sensor=CONFIG["sensor"],
        slice_mode=CONFIG["temporal_mode"],
        transform=train_transform,
    )
    val_ds = TimeSeriesDataset(
        val_ref_ids,
        sensor=CONFIG["sensor"],
        slice_mode=CONFIG["temporal_mode"],
        transform=val_transform,
    )
    test_ds = TimeSeriesDataset(
        test_ref_ids,
        sensor=CONFIG["sensor"],
        slice_mode=CONFIG["temporal_mode"],
        transform=test_transform,
    )
    
    print(f"✓ Datasets created for pre-cropped {CONFIG['chip_size']}×{CONFIG['chip_size']} chips")
    print(f"Train chips: {len(train_ds)} (from {len(train_ref_ids)} REFIDs) - with flips + rotations")
    print(f"Val chips: {len(val_ds)} (from {len(val_ref_ids)} REFIDs) - no augmentation")
    print(f"Test chips: {len(test_ds)} (from {len(test_ref_ids)} REFIDs) - no augmentation")
    print(f"Augmentation enabled: {CONFIG['augment_train']}")
    
    # Create dataloaders
    def worker_init_fn(worker_id):
        worker_seed = CONFIG["random_seed"] + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(CONFIG["random_seed"])
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,  # Use batch_size=1 for stable validation on small datasets
        shuffle=False,
        num_workers=CONFIG["num_workers"]
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,  # Use batch_size=1 for stable test evaluation
        shuffle=False,
        num_workers=CONFIG["num_workers"]
    )
    
    # Build model
    print("\n" + "="*80)
    print("MODEL")
    print("="*80)
    
    # Get sample batch to determine input shape
    sample_x, _ = next(iter(train_loader))
    _, T, C, H, W = sample_x.shape
    
    # Early fusion: T * C channels
    num_input_channels = T * C
    
    model = smp.Unet(
        encoder_name=CONFIG["encoder_name"],
        encoder_weights=CONFIG["encoder_weights"],
        in_channels=num_input_channels,
        classes=CONFIG["num_classes"]
    ).to(device)
    
    print(f"✓ U-Net model created with early fusion")
    print(f"  Timesteps: {T}")
    print(f"  Channels per timestep: {C}")
    print(f"  Total input channels (T*C): {num_input_channels}")
    print(f"  Encoder: {CONFIG['encoder_name']}")
    print(f"  Encoder weights: {CONFIG['encoder_weights']}")
    print(f"  Classes: {CONFIG['num_classes']}")
    
    # Loss, optimizer, and scaler
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scaler = torch.cuda.amp.GradScaler()
    
    # Initialize WandB
    print("\n" + "="*80)
    print("WANDB INITIALIZATION")
    print("="*80)
    run = wandb.init(
        project=CONFIG["wandb_project"],
        entity=CONFIG["wandb_entity"],
        name=f"UNet_EarlyFusion_{CONFIG['sensor']}_chip{CONFIG['chip_size']}_t{T}",
        config={
            "model": "Unet_EarlyFusion",
            "architecture": CONFIG["architecture"],
            "encoder": CONFIG["encoder_name"],
            "encoder_weights": CONFIG["encoder_weights"],
            "in_channels": num_input_channels,
            "timesteps": T,
            "channels_per_timestep": C,
            "classes": CONFIG["num_classes"],
            "learning_rate": CONFIG["learning_rate"],
            "batch_size": CONFIG["batch_size"],
            "chip_size": CONFIG["chip_size"],
            "epochs": CONFIG["epochs"],
            "augment_train": CONFIG["augment_train"],
            "augmentation": "flips_rotations" if CONFIG["augment_train"] else "none",
            "temporal_mode": CONFIG["temporal_mode"],
            "sensor": CONFIG["sensor"],
            "train_chips": len(train_ds),
            "val_chips": len(val_ds),
            "test_chips": len(test_ds),
            "normalization": CONFIG["normalization"],
            "random_seed": CONFIG["random_seed"],
            "train_ratio": CONFIG["train_ratio"],
            "val_ratio": CONFIG["val_ratio"],
            "test_ratio": CONFIG["test_ratio"],
            "preprocessing": "64x64_chips_no_patching",
        },
    )
    
    print("✓ WandB initialized")
    
    # Training loop
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    for epoch in range(CONFIG["epochs"]):
        # Training
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device, scaler)
        
        # Validation
        val_loss, val_metrics = validate(model, val_loader, loss_fn, device)
        
        # Log to WandB
        run.log({
            "epoch": epoch + 1,
            "avg_train_loss": train_loss,
            "avg_val_loss": val_loss,
            "IoU": val_metrics['iou'],
            "F1": val_metrics['f1'],
            "Precision": val_metrics['precision'],
            "Recall": val_metrics['recall'],
            "Accuracy": val_metrics['accuracy']
        })
        
        # Print epoch summary
        print(
            f"Epoch {epoch+1}/{CONFIG['epochs']}: "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} | "
            f"IoU={val_metrics['iou']:.4f} "
            f"F1={val_metrics['f1']:.4f} "
            f"Prec={val_metrics['precision']:.4f} "
            f"Rec={val_metrics['recall']:.4f} "
            f"Acc={val_metrics['accuracy']:.4f}"
        )
    
    # Test set evaluation
    print("\n" + "="*80)
    print("TEST SET EVALUATION")
    print("="*80)
    test_loss, test_metrics = validate(model, test_loader, loss_fn, device)
    
    print(f"Test Set Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  IoU: {test_metrics['iou']:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    
    # Log test metrics to WandB
    run.log({
        "test_loss": test_loss,
        "test_iou": test_metrics['iou'],
        "test_f1": test_metrics['f1'],
        "test_precision": test_metrics['precision'],
        "test_recall": test_metrics['recall'],
        "test_accuracy": test_metrics['accuracy'],
    })
    
    # Log combined masks from multiple test batches
    print("\nLogging test set masks...")
    log_masks(model, test_loader, device, step=CONFIG["epochs"], name_prefix="test", max_batches=10)
    
    # Finish WandB
    run.finish()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Final Validation Metrics:")
    print(f"  Loss: {val_loss:.4f}")
    print(f"  IoU: {val_metrics['iou']:.4f}")
    print(f"  F1: {val_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
