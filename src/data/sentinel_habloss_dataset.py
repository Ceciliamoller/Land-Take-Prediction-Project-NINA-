"""
Multitemporal Sentinel HABLOSS dataset for land-take segmentation.
"""

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import rasterio


class SentinelHablossPatchDataset(Dataset):
    """
    PyTorch Dataset for multitemporal Sentinel imagery and land-take masks.
    
    Loads multitemporal Sentinel images (Loads full 126-band Sentinel-2 temporal-spectral stacks.)
    and corresponding binary land-take masks. Returns random patches for training.
    
    The dataset handles:
    - Automatic pairing of Sentinel images with their corresponding masks
    - Mask upsampling to match Sentinel resolution
    - Random patch extraction from full tiles
    - Per-channel normalization (standardization or simple scaling)
    - Optional data augmentation (random flips)
    """
    
    def __init__(
        self,
        sentinel_dir: Union[str, Path],
        mask_dir: Union[str, Path],
        patch_size: int = 64,
        patches_per_image: int = 10,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
        augment: bool = False,
        ref_ids: Optional[list[str]] = None,
    ):
        """
        Initialize the dataset by finding matching Sentinel-mask pairs.
        
        Args:
            sentinel_dir: Directory containing Sentinel *_RGBNIRRSWIRQ_Mosaic.tif files
            mask_dir: Directory containing *_mask.tif files
            patch_size: Size of square patches (e.g., 64)
            patches_per_image: Number of patches to sample per tile per epoch
            mean: Optional sequence for per-channel mean normalization (length = total bands)
            std: Optional sequence for per-channel std normalization (length = total bands)
            augment: Whether to apply random flips for data augmentation
            ref_ids: Optional list of REFIDs to include. If None, uses all matching pairs.
        
        Raises:
            RuntimeError: If no valid Sentinel-mask pairs are found
        """
        self.sentinel_dir = Path(sentinel_dir)
        self.mask_dir = Path(mask_dir)
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.augment = augment
        self.ref_ids = set(ref_ids) if ref_ids is not None else None
        
        self.pairs = self._find_pairs()
        
        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No valid Sentinel-mask pairs found in {self.sentinel_dir} and {self.mask_dir}"
            )
        
        with rasterio.open(self.pairs[0][0]) as src:
            self.num_bands = src.count
        
        if mean is not None and std is not None:
            self.mean = torch.tensor(mean, dtype=torch.float32).view(self.num_bands, 1, 1)
            self.std = torch.tensor(std, dtype=torch.float32).view(self.num_bands, 1, 1)
        else:
            self.mean = None
            self.std = None
    
    def _find_pairs(self) -> list:
        """
        Scan directories and pair Sentinel images with their corresponding masks.
        Filters by ref_ids if provided.
        
        Returns:
            List of (sentinel_path, mask_path) tuples
        """
        pairs = []
        sentinel_files = sorted(self.sentinel_dir.glob("*_RGBNIRRSWIRQ_Mosaic.tif"))
        
        for sentinel_path in sentinel_files:
            refid = sentinel_path.stem.replace("_RGBNIRRSWIRQ_Mosaic", "")
            
            if self.ref_ids is not None and refid not in self.ref_ids:
                continue
            
            mask_path = self.mask_dir / f"{refid}_mask.tif"
            
            if mask_path.exists():
                pairs.append((sentinel_path, mask_path))
        
        return pairs
    
    def __len__(self) -> int:
        """Total number of patches per epoch."""
        return len(self.pairs) * self.patches_per_image
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and return a random patch from a Sentinel-mask pair.
        
        Args:
            idx: Index for the patch
        
        Returns:
            img_patch: torch.float32, shape (num_bands, patch_size, patch_size)
            mask_patch: torch.int64, shape (patch_size, patch_size)
        """
        pair_idx = idx // self.patches_per_image
        sentinel_path, mask_path = self.pairs[pair_idx]
        
        with rasterio.open(sentinel_path) as src:
            img = src.read()
        
        with rasterio.open(mask_path) as src_m:
            mask = src_m.read(1)
        
        # Replace NaN values with 0 (no-data areas in satellite imagery)
        img = np.nan_to_num(img, nan=0.0)
        
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).long()
        
        C, H, W = img.shape
        h, w = mask.shape
        
        if h != H or w != W:
            mask_up = mask.unsqueeze(0).unsqueeze(0).float()
            mask_up = F.interpolate(mask_up, size=(H, W), mode="nearest")
            mask = mask_up.squeeze(0).squeeze(0).long()
        
        if H < self.patch_size or W < self.patch_size:
            pad_h = max(0, self.patch_size - H)
            pad_w = max(0, self.patch_size - W)
            img = F.pad(img, (0, pad_w, 0, pad_h), mode="constant", value=0)
            mask = F.pad(mask, (0, pad_w, 0, pad_h), mode="constant", value=0)
            C, H, W = img.shape
        
        max_y = H - self.patch_size
        max_x = W - self.patch_size
        y = random.randint(0, max_y)
        x = random.randint(0, max_x)
        
        img_patch = img[:, y:y+self.patch_size, x:x+self.patch_size]
        mask_patch = mask[y:y+self.patch_size, x:x+self.patch_size]
        
        img_patch = img_patch / 10000.0
        
        if self.mean is not None and self.std is not None:
            img_patch = (img_patch - self.mean) / (self.std + 1e-6)
        
        if self.augment:
            if random.random() > 0.5:
                img_patch = torch.flip(img_patch, dims=[2])
                mask_patch = torch.flip(mask_patch, dims=[1])
            
            if random.random() > 0.5:
                img_patch = torch.flip(img_patch, dims=[1])
                mask_patch = torch.flip(mask_patch, dims=[0])
        
        return img_patch, mask_patch


def estimate_mean_std(
    dataset: SentinelHablossPatchDataset,
    num_samples: int = 2000,
) -> Tuple[list, list]:
    """
    Estimate per-channel mean and std from a sample of patches.
    
    Args:
        dataset: SentinelHablossPatchDataset instance (should have mean/std=None)
        num_samples: Number of random patches to sample for statistics
    
    Returns:
        mean: List of mean values (one per channel)
        std: List of std values (one per channel)
    """
    num_samples = min(num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), num_samples)
    
    patches = []
    for idx in indices:
        img_patch, _ = dataset[idx]
        patches.append(img_patch)
    
    all_patches = torch.stack(patches, dim=0)
    
    mean = all_patches.mean(dim=[0, 2, 3]).tolist()
    std = all_patches.std(dim=[0, 2, 3]).tolist()
    
    return mean, std
