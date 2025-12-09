"""
FCEF Early Fusion Training Script for Land-Take Prediction

Based on fc_early_fusion.ipynb
Fair comparison setup with U-Net baseline: shared splits, normalization, patch size, random seeds
"""

import sys
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import wandb

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
from src.models.external.torchrs_fc_cd import FCEF

import wandb

def log_example_batch(model, loader, device, step, name_prefix="val"):
    model.eval()
    imgs, masks = next(iter(loader))        # imgs shape: (B, T, C, H, W)
    
    with torch.no_grad():
        B, T, C, H, W = imgs.shape
        # FCEF expects (B, T, C, H, W)
        x = imgs.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu()  # (B, H, W)

    masks = masks.cpu()

    # Make a simple RGB image from first timestep (bands 0,1,2)
    rgb = imgs[:, 0, :3, :, :].cpu()        # (B, 3, H, W)
    # Re-stretch RGB to [0, 1] for better visualization
    rgb_min = rgb.amin(dim=(-2, -1), keepdim=True)
    rgb_max = rgb.amax(dim=(-2, -1), keepdim=True)
    rgb = (rgb - rgb_min) / (rgb_max - rgb_min + 1e-6)

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


def log_masks(model, loader, device, step, name_prefix="val", max_batches=10):
    """
    Log ground-truth and predicted segmentation masks to WandB.
    Iterates over multiple batches and logs GT and predictions separately.
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
        model.eval()
        gt_images = []
        pred_images = []
        
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
                
                B = imgs.shape[0]
                x = imgs.to(device)
                logits = model(x)
                preds = logits.argmax(dim=1).cpu()  # (B, H, W)
                masks = masks.cpu()
                
                # Convert to uint8 and scale to 0/255 for visibility
                masks_vis = (masks * 255).byte().numpy()  # (B, H, W)
                preds_vis = (preds * 255).byte().numpy()  # (B, H, W)
                
                for i in range(B):
                    if len(masks_vis[i].shape) != 2 or len(preds_vis[i].shape) != 2:
                        continue
                    
                    gt_images.append(wandb.Image(masks_vis[i], caption=f"{name_prefix}_gt_b{b_idx}_i{i}"))
                    pred_images.append(wandb.Image(preds_vis[i], caption=f"{name_prefix}_pred_b{b_idx}_i{i}"))
        
        # Log ground truth and predictions separately
        if len(gt_images) > 0 and len(pred_images) > 0:
            wandb.log({f"{name_prefix}_gt_masks": gt_images}, step=step)
            wandb.log({f"{name_prefix}_pred_masks": pred_images}, step=step)
            print(f"[INFO] Logged {len(gt_images)} GT masks and {len(pred_images)} prediction masks for {name_prefix}")
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
    "architecture": "FCEF",
    "num_classes": 2,
    
    # Data
    "sensor": "sentinel",
    "temporal_mode": "first_half",  # 7 timesteps
    "chip_size": 64,  # Pre-cropped chips are 64×64
    
    # Training
    "epochs": 50,
    "learning_rate": 1e-3,
    "batch_size": 4,
    "augment_train": True,  # Enable spatial augmentation (flips, rotations)
    
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
    print(f"✓ Using SHARED splits with U-Net baseline (random_state={CONFIG['random_seed']})")
    
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
        batch_size=1,  # Use batch_size=1 for stable validation on small datasets
        shuffle=False,
        num_workers=CONFIG["num_workers"],
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,  # Use batch_size=1 for stable test evaluation
        shuffle=False,
        num_workers=CONFIG["num_workers"],
    )
    
    print(f"✓ Dataloaders created with reproducible shuffling (seed={CONFIG['random_seed']})")
    
    # Build model
    print("\n" + "="*80)
    print("MODEL")
    print("="*80)
    sample_x, _ = next(iter(train_loader))
    _, T, C, H, W = sample_x.shape
    
    model = FCEF(channels=C, t=T, num_classes=CONFIG["num_classes"]).to(device)
    
    print(f"✓ FCEF model created")
    print(f"  Channels: {C}")
    print(f"  Timesteps: {T}")
    print(f"  Classes: {CONFIG['num_classes']}")
    print(f"  Input shape: (B, {T}, {C}, {H}, {W})")
    
    # Loss, optimizer, and scaler
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scaler = torch.cuda.amp.GradScaler()
    
    # Initialize WandB
    print("\n" + "="*80)
    print("WANDB INITIALIZATION")
    print("="*80)
    run = wandb.init(
        entity=CONFIG["wandb_entity"],
        project=CONFIG["wandb_project"],
        name=f"FCEF_{CONFIG['sensor']}_chip{CONFIG['chip_size']}_t{T}",
        config={
            "learning_rate": CONFIG["learning_rate"],
            "architecture": CONFIG["architecture"],
            "dataset": CONFIG["sensor"],
            "epochs": CONFIG["epochs"],
            "batch_size": CONFIG["batch_size"],
            "chip_size": CONFIG["chip_size"],
            "augment_train": CONFIG["augment_train"],
            "augmentation": "flips_rotations" if CONFIG["augment_train"] else "none",
            "temporal_mode": CONFIG["temporal_mode"],
            "num_timesteps": T,
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
        model.train()
        total_loss = 0.0
        for x, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}"):
            x = x.to(device)
            mask = mask.to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits = model(x)
                loss = criterion(logits, mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        sum_tp = sum_fp = sum_tn = sum_fn = 0
        with torch.no_grad():
            for x, mask in val_loader:
                x = x.to(device)
                mask = mask.to(device)
                with torch.amp.autocast('cuda'):
                    logits = model(x)
                    loss = criterion(logits, mask)
                val_loss += loss.item()

                pred = torch.argmax(logits, dim=1)
                tp, fp, tn, fn = compute_confusion_binary(pred, mask, positive_class=1)
                sum_tp += tp
                sum_fp += fp
                sum_tn += tn
                sum_fn += fn

        avg_val_loss = val_loss / len(val_loader)
        val_metrics = compute_metrics_from_confusion(sum_tp, sum_fp, sum_tn, sum_fn)

        # Log to WandB
        run.log({
            "epoch": epoch + 1,
            "avg_train_loss": avg_train_loss,
            "avg_val_loss": avg_val_loss,
            "IoU": val_metrics['iou'],
            "F1": val_metrics['f1'],
            "Precision": val_metrics['precision'],
            "Recall": val_metrics['recall'],
            "Accuracy": val_metrics['accuracy']
        })

        # Print epoch summary
        print(
            f"Epoch {epoch+1}/{CONFIG['epochs']}: "
            f"train_loss={avg_train_loss:.4f} "
            f"val_loss={avg_val_loss:.4f} | "
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
    model.eval()
    test_loss = 0.0
    sum_tp = sum_fp = sum_tn = sum_fn = 0
    
    with torch.no_grad():
        for x, mask in test_loader:
            x = x.to(device)
            mask = mask.to(device)
            with torch.amp.autocast('cuda'):
                logits = model(x)
                loss = criterion(logits, mask)
            test_loss += loss.item()

            pred = torch.argmax(logits, dim=1)
            tp, fp, tn, fn = compute_confusion_binary(pred, mask, positive_class=1)
            sum_tp += tp
            sum_fp += fp
            sum_tn += tn
            sum_fn += fn

    avg_test_loss = test_loss / len(test_loader)
    test_metrics = compute_metrics_from_confusion(sum_tp, sum_fp, sum_tn, sum_fn)
    
    print(f"Test Set Results:")
    print(f"  Loss: {avg_test_loss:.4f}")
    print(f"  IoU: {test_metrics['iou']:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    
    # Log test metrics to WandB
    run.log({
        "test_loss": avg_test_loss,
        "test_iou": test_metrics['iou'],
        "test_f1": test_metrics['f1'],
        "test_precision": test_metrics['precision'],
        "test_recall": test_metrics['recall'],
        "test_accuracy": test_metrics['accuracy'],
    })
    
    # Always log example predictions from test set at the end
    print("\nLogging final test set predictions...")
    log_example_batch(model, test_loader, device, step=CONFIG["epochs"], name_prefix="test")
    
    # Log masks from multiple test batches
    print("\nLogging test set masks...")
    log_masks(model, test_loader, device, step=CONFIG["epochs"], name_prefix="test", max_batches=10)
    
    # Finish WandB
    run.finish()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Final Validation Metrics:")
    print(f"  Loss: {avg_val_loss:.4f}")
    print(f"  IoU: {val_metrics['iou']:.4f}")
    print(f"  F1: {val_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
