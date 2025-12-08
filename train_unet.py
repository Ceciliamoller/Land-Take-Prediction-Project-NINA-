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

def log_example_batch(model, loader, device, step, name_prefix="val"):
    model.eval()
    imgs, masks = next(iter(loader))        # imgs shape:
                                            # UNet EF: (B, T, C, H, W)
                                            # FCEF:    (B, T, C, H, W)
    with torch.no_grad():
        B, T, C, H, W = imgs.shape
        # For U-Net early fusion, flatten T and C
        x_unet = imgs.reshape(B, T * C, H, W).to(device)
        logits = model(x_unet)
        preds = logits.argmax(dim=1).cpu()  # (B, H, W)

    masks = masks.cpu()

    # Make a simple RGB image from first timestep (bands 0,1,2)
    rgb = imgs[:, 0, :3, :, :].cpu()        # (B, 3, H, W)

    wandb_images = []
    for i in range(min(4, B)):
        wandb_images.append(
            wandb.Image(
                rgb[i],
                masks={
                    "ground_truth": {
                        "mask_data": masks[i].numpy(),
                        "class_labels": {0: "background", 1: "land-take"},
                    },
                    "prediction": {
                        "mask_data": preds[i].numpy(),
                        "class_labels": {0: "background", 1: "land-take"},
                    },
                },
            )
        )

    wandb.log({f"{name_prefix}_examples": wandb_images}, step=step)

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
            RandomFlipTS(p_horizontal=0.5, p_vertical=0.5),
            RandomRotate90TS(),
            NormalizeBy(10000.0),
            Normalize(mean, std),
        ])
    else:
        train_transform = ComposeTS([
            NormalizeBy(10000.0),
            Normalize(mean, std),
        ])
    
    # Val/test transforms: no augmentation, only normalization
    val_transform = ComposeTS([
        NormalizeBy(10000.0),
        Normalize(mean, std),
    ])
    
    test_transform = ComposeTS([
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
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"]
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=CONFIG["batch_size"],
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

        if epoch % 5 == 0:
            log_example_batch(model, val_loader, device, step=epoch, name_prefix="val")
    
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
