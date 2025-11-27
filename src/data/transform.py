"""
Shared data transformation utilities for fair model comparison.

This module provides consistent normalization and augmentation transforms
for both U-Net and FCEF baselines, ensuring identical preprocessing.
"""

import torch
import random
import numpy as np
from typing import Tuple, Optional, Sequence


def compute_normalization_stats(
    dataset,
    num_samples: int = 2000,
) -> Tuple[list[float], list[float]]:
    """
    Compute per-channel mean and std from training dataset samples.
    
    This function estimates normalization statistics from a random subset of
    training patches. These statistics should be computed ONCE from the training
    set and then applied consistently to train/val/test data.
    
    Args:
        dataset: PyTorch dataset that returns (image, mask) tuples
                 Image should already be scaled (e.g., divided by 10000)
        num_samples: Number of random samples to use for estimation
    
    Returns:
        Tuple of (mean_list, std_list) with one value per channel
    
    Example:
        >>> train_ds = SentinelHablossPatchDataset(...)  # with mean=None, std=None
        >>> mean, std = compute_normalization_stats(train_ds, num_samples=2000)
        >>> # Now use these mean/std for train_ds, val_ds, and test_ds
    """
    num_samples = min(num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), num_samples)
    
    patches = []
    for idx in indices:
        img_patch, _ = dataset[idx]
        patches.append(img_patch)
    
    all_patches = torch.stack(patches, dim=0)  # (N, C, H, W)
    
    # Compute mean and std across all spatial and sample dimensions
    mean = all_patches.mean(dim=[0, 2, 3]).tolist()
    std = all_patches.std(dim=[0, 2, 3]).tolist()
    
    return mean, std


class Normalize:
    """
    Apply per-channel standardization using precomputed mean and std.
    
    This transform should be applied AFTER scaling (e.g., dividing by 10000).
    Use the same mean/std values computed from the training set for all splits.
    
    Args:
        mean: Sequence of per-channel mean values
        std: Sequence of per-channel std values
        
    Example:
        >>> # For flattened Sentinel data (B, C, H, W)
        >>> transform = Normalize(mean=train_mean, std=train_std)
        >>> img_normalized, mask = transform(img_scaled, mask)
    """
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
    
    def __call__(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Handle different input shapes
        if x.dim() == 4:  # (T, C, H, W) for time series
            T, C, H, W = x.shape
            mean = self.mean.view(1, C, 1, 1)
            std = self.std.view(1, C, 1, 1)
        elif x.dim() == 3:  # (C, H, W) for standard images
            C, H, W = x.shape
            mean = self.mean.view(C, 1, 1)
            std = self.std.view(C, 1, 1)
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got shape {x.shape}")
        
        x = (x - mean) / (std + 1e-6)
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        return x, mask


class RandomCropTS:
    """returns cropped image + mask with default size H,W => 64,64"""

    def __init__(self, size=64):
        self.size = size

    def __call__(self, x, mask):
        # x: (T, C, H, W)
        T, C, H, W = x.shape
        s = self.size

        top = 0 if H <= s else random.randint(0, H - s)
        left = 0 if W <= s else random.randint(0, W - s)

        x = x[:, :, top:top+s, left:left+s]
        mask = mask[top:top+s, left:left+s]

        return x, mask 
    
class CenterCropTS:
    def __init__(self, size):
        self.size = size 
    def __call__(self, x, mask):
        T, C, H, W = x.shape
        s = self.size
        top = max(0, (H - s) // 2)
        left = max(0, (W - s) // 2)
        x = x[:, :, top:top+s, left:left+s]
        mask = mask[top:top+s, left:left+s]
        return x, mask
    
class NormalizeBy:
    """Divide by a constant (Sentinel is TOA×10000)."""
    def __init__(self, denom=10000.0):
        self.denom = denom
    def __call__(self, x, mask):
        x = x / self.denom
        # replace NaNs and infinities with 0 (or some sensible default)
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        return x, mask


class ComposeTS:
    def __init__(self, ops):
        self.ops = ops
    def __call__(self, x, mask):
        for op in self.ops:
            x, mask = op(x, mask)
        return x, mask