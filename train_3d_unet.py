"""
U-Net Training Script for Land-Take Prediction

Based on 03_smp_unet_baseline.ipynb
Fair comparison setup with FCEF baseline: shared splits, normalization, patch size, random seeds
"""

import sys
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
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
    RandomRotate90TS,
)
from src.models.three_d_unet import ThreeDUNet


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
    "encoder_name": "resnet34",
    "encoder_weights": "imagenet",
    "num_classes": 2,
    # TimeSeries options (for 3D-UNet input via TimeSeriesDataset)
    "sensor": "sentinel",
    "temporal_mode": "first_half",
    
    # Training
    "epochs": 50,
    "learning_rate": 1e-3,
    "batch_size": 8,
    "patch_size": 64,
    "patches_per_image_train": 20,
    "patches_per_image_val": 10,
    "patches_per_image_test": 10,
    "augment_train": True,
    
    # Normalization
    "normalization": "scale_10000_plus_standardize",
    "num_samples_for_stats": 2000,
    
    # DataLoader
    "num_workers": 0,
    
    # WandB
    "wandb_project": "Baseline",
    "wandb_entity": "nina_prosjektoppgave",
    
    # Logging
    "log_examples_every_n_epochs": 2,
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

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0

    for imgs, masks in loader:
        # imgs from TimeSeriesDataset: (B, T, C, H, W)
        imgs = imgs.to(device)
        # Convert to (B, C, D, H, W) for Conv3d-based model
        imgs = imgs.permute(0, 2, 1, 3, 4).contiguous()
        masks = masks.to(device)

        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            logits = model(imgs)
            loss = loss_fn(logits, masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss


def validate(model, loader, loss_fn, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    sum_tp = sum_fp = sum_tn = sum_fn = 0

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            imgs = imgs.permute(0, 2, 1, 3, 4).contiguous()
            masks = masks.to(device)

            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                logits = model(imgs)
                loss = loss_fn(logits, masks)
            
            total_loss += loss.item() * imgs.size(0)

            pred = torch.argmax(logits, dim=1)
            tp, fp, tn, fn = compute_confusion_binary(pred, masks, positive_class=1)
            sum_tp += tp
            sum_fp += fp
            sum_tn += tn
            sum_fn += fn

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics_from_confusion(sum_tp, sum_fp, sum_tn, sum_fn)
    
    return avg_loss, metrics


def log_examples(model, loader, device, step, phase="val"):
    """Log example predictions to wandb"""
    model.eval()
    with torch.no_grad():
        imgs, masks = next(iter(loader))
        # imgs: (B, T, C, H, W)
        imgs_device = imgs.to(device)
        imgs_for_model = imgs_device.permute(0, 2, 1, 3, 4).contiguous()  # (B, C, D, H, W)
        preds = model(imgs_for_model)

        preds_class = preds.argmax(dim=1)

        # Use first timestep RGB for visualization: imgs[:, 0, C, H, W] -> (B, C, H, W)
        rgb_imgs = imgs[:, 0, :3, :, :].clone()
        for i in range(3):
            min_val = rgb_imgs[:, i, :, :].min()
            max_val = rgb_imgs[:, i, :, :].max()
            if max_val > min_val:
                rgb_imgs[:, i, :, :] = (rgb_imgs[:, i, :, :] - min_val) / (max_val - min_val)
        
        wandb_images = []
        for i in range(min(4, imgs.size(0))):
            wandb_images.append(
                wandb.Image(
                    rgb_imgs[i].cpu(),
                    masks={
                        "ground_truth": {
                            "mask_data": masks[i].cpu().numpy(),
                            "class_labels": {0: "background", 1: "land-take"}
                        },
                        "prediction": {
                            "mask_data": preds_class[i].cpu().numpy(),
                            "class_labels": {0: "background", 1: "land-take"}
                        },
                    },
                )
            )
        
        wandb.log({f"{phase}_examples": wandb_images}, step=step)


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
    
    # Compute normalization stats for time-series (first half)
    print("\n" + "="*80)
    print("NORMALIZATION")
    print("="*80)
    temp_train_transform = ComposeTS([
        NormalizeBy(10000.0),
        CenterCropTS(CONFIG["patch_size"]),
    ])

    temp_train_ds = TimeSeriesDataset(
        train_ref_ids,
        sensor=CONFIG["sensor"],
        slice_mode=CONFIG["temporal_mode"],
        transform=temp_train_transform,
        patches_per_image=5,
    )

    print("Estimating per-channel mean and std from training data...")
    mean, std = compute_normalization_stats(temp_train_ds, num_samples=CONFIG["num_samples_for_stats"])
    print(f"✓ Computed normalization stats: {len(mean)} channels")
    print(f"  Mean (first 5): {[f'{m:.4f}' for m in mean[:5]]}")
    print(f"  Std (first 5): {[f'{s:.4f}' for s in std[:5]]}")

    # Create datasets (time series mode: first half of temporal stack)
    print("\n" + "="*80)
    print("DATASETS")
    print("="*80)

    train_transform_ops = [
        NormalizeBy(10000.0),
        Normalize(mean, std),
        RandomCropTS(CONFIG["patch_size"]),
    ]
    if CONFIG["augment_train"]:
        train_transform_ops.extend([
            RandomFlipTS(p_horizontal=0.5, p_vertical=0.5),
            RandomRotate90TS(p=0.5),
        ])
    train_transform = ComposeTS(train_transform_ops)

    val_transform = ComposeTS([
        NormalizeBy(10000.0),
        Normalize(mean, std),
        CenterCropTS(CONFIG["patch_size"]),
    ])

    test_transform = ComposeTS([
        NormalizeBy(10000.0),
        Normalize(mean, std),
        CenterCropTS(CONFIG["patch_size"]),
    ])

    train_ds = TimeSeriesDataset(
        train_ref_ids,
        sensor=CONFIG["sensor"],
        slice_mode=CONFIG["temporal_mode"],
        transform=train_transform,
        patches_per_image=CONFIG["patches_per_image_train"],
    )
    val_ds = TimeSeriesDataset(
        val_ref_ids,
        sensor=CONFIG["sensor"],
        slice_mode=CONFIG["temporal_mode"],
        transform=val_transform,
        patches_per_image=CONFIG["patches_per_image_val"],
    )
    test_ds = TimeSeriesDataset(
        test_ref_ids,
        sensor=CONFIG["sensor"],
        slice_mode=CONFIG["temporal_mode"],
        transform=test_transform,
        patches_per_image=CONFIG["patches_per_image_test"],
    )

    print(f"✓ Datasets created with SHARED normalization and patch_size={CONFIG['patch_size']}")
    print(f"Training patches: {len(train_ds)} (from {len(train_ref_ids)} tiles, {CONFIG['patches_per_image_train']} patches/tile) - random crops + augmentation")
    print(f"Validation patches: {len(val_ds)} (from {len(val_ref_ids)} tiles, {CONFIG['patches_per_image_val']} patches/tile) - deterministic center crops")
    print(f"Test patches: {len(test_ds)} (from {len(test_ref_ids)} tiles, {CONFIG['patches_per_image_test']} patches/tile) - deterministic center crops")
    
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
    # Build 3D U-Net model
    print("\n" + "="*80)
    print("MODEL")
    print("="*80)

    # Peek a sample to get channels/timesteps
    sample_x, _ = next(iter(train_loader))
    # sample_x: (B, T, C, H, W)
    B, T, C, H, W = sample_x.shape
    print(f"Data sample shape (B, T, C, H, W): {sample_x.shape}")

    model = ThreeDUNet(in_channels=C, num_classes=CONFIG["num_classes"]).to(device)

    print(f"✓ 3D U-Net model created")
    print(f"  Input channels per timestep: {C}")
    print(f"  Timesteps (depth): {T}")
    print(f"  Classes: {CONFIG['num_classes']}")

    # Loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    # Initialize WandB
    print("\n" + "="*80)
    print("WANDB INITIALIZATION")
    print("="*80)
    wandb.init(
        project=CONFIG["wandb_project"],
        entity=CONFIG["wandb_entity"],
        name=f"3DUNet_patch{CONFIG['patch_size']}_t{T}",
        config={
            "model": "3D-UNet",
            "in_channels": C,
            "timesteps": T,
            "classes": CONFIG["num_classes"],
            "learning_rate": CONFIG["learning_rate"],
            "batch_size": CONFIG["batch_size"],
            "patch_size": CONFIG["patch_size"],
            "epochs": CONFIG["epochs"],
            "train_patches_per_image": CONFIG["patches_per_image_train"],
            "val_patches_per_image": CONFIG["patches_per_image_val"],
            "test_patches_per_image": CONFIG["patches_per_image_test"],
            "train_ref_ids": len(train_ref_ids),
            "val_ref_ids": len(val_ref_ids),
            "test_ref_ids": len(test_ref_ids),
            "augmentation": CONFIG["augment_train"],
            "normalization": CONFIG["normalization"],
            "random_seed": CONFIG["random_seed"],
            "train_ratio": CONFIG["train_ratio"],
            "val_ratio": CONFIG["val_ratio"],
            "test_ratio": CONFIG["test_ratio"],
            "fair_comparison": "shared_splits_normalization_patch_size_with_FCEF",
        },
    )

    wandb.watch(model, log="all", log_freq=100)
    print("✓ WandB initialized")
    
    # Training loop
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    train_losses = []
    val_losses = []
    val_ious = []
    val_f1s = []
    
    for epoch in range(CONFIG["epochs"]):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        train_losses.append(train_loss)
        
        val_loss, val_metrics = validate(model, val_loader, loss_fn, device)
        val_losses.append(val_loss)
        val_ious.append(val_metrics['iou'])
        val_f1s.append(val_metrics['f1'])
        
        # Log to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_iou": val_metrics['iou'],
            "val_f1": val_metrics['f1'],
            "val_precision": val_metrics['precision'],
            "val_recall": val_metrics['recall'],
            "val_accuracy": val_metrics['accuracy'],
        })
        
        # Log example predictions
        if (epoch + 1) % CONFIG["log_examples_every_n_epochs"] == 0:
            log_examples(model, val_loader, device, step=epoch + 1, phase="val")
        
        # Print epoch summary
        print(
            f"Epoch {epoch+1}/{CONFIG['epochs']}: "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} | "
            f"IoU={val_metrics['iou']:.4f} "
            f"F1={val_metrics['f1']:.4f} "
            f"Prec={val_metrics['precision']:.4f} "
            f"Rec={val_metrics['recall']:.4f}"
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
    wandb.log({
        "test_loss": test_loss,
        "test_iou": test_metrics['iou'],
        "test_f1": test_metrics['f1'],
        "test_precision": test_metrics['precision'],
        "test_recall": test_metrics['recall'],
        "test_accuracy": test_metrics['accuracy'],
    })
    
    # Finish WandB
    wandb.finish()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Final Validation Metrics:")
    print(f"  Loss: {val_losses[-1]:.4f}")
    print(f"  IoU: {val_ious[-1]:.4f}")
    print(f"  F1: {val_f1s[-1]:.4f}")


if __name__ == "__main__":
    main()
