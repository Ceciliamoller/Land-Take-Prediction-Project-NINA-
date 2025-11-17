"""
Random HABLOSS VHR+mask patches for segmentation training.
"""

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import rasterio


class SimpleHablossPatchDataset(Dataset):
    """
    PyTorch Dataset for HABLOSS VHR imagery and land-take masks.
    
    Loads paired before/after satellite images (6-band: RGB before + RGB after)
    and corresponding binary land-take masks. Returns random patches for training.
    
    The dataset handles:
    - Automatic pairing of VHR images with their corresponding masks
    - Mask upsampling to match VHR resolution
    - Random patch extraction from full tiles
    - Per-channel normalization (standardization or simple scaling)
    - Optional data augmentation (random flips)
    """
    
    def __init__(
        self,
        vhr_dir: Union[str, Path],
        mask_dir: Union[str, Path],
        patch_size: int = 128,
        patches_per_image: int = 10,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
        augment: bool = False,
    ):
        """
        Initialize the dataset by finding matching VHR-mask pairs.
        
        Args:
            vhr_dir: Directory containing VHR *_RGBY_Mosaic.tif files
            mask_dir: Directory containing *_mask.tif files
            patch_size: Size of square patches (e.g., 128)
            patches_per_image: Number of patches to sample per tile per epoch
            mean: Optional 6-element sequence for per-channel mean normalization
            std: Optional 6-element sequence for per-channel std normalization
            augment: Whether to apply random flips for data augmentation
        
        Raises:
            RuntimeError: If no valid VHR-mask pairs are found
        """
        self.vhr_dir = Path(vhr_dir)
        self.mask_dir = Path(mask_dir)
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.augment = augment
        
        if mean is not None and std is not None:
            self.mean = torch.tensor(mean, dtype=torch.float32).view(6, 1, 1)
            self.std = torch.tensor(std, dtype=torch.float32).view(6, 1, 1)
        else:
            self.mean = None
            self.std = None
        
        self.pairs = self._find_pairs()
        
        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No valid VHR-mask pairs found in {self.vhr_dir} and {self.mask_dir}"
            )
    
    def _find_pairs(self) -> list:
        """
        Scan directories and pair VHR images with their corresponding masks.
        
        Returns:
            List of (vhr_path, mask_path) tuples
        """
        pairs = []
        vhr_files = sorted(self.vhr_dir.glob("*_RGBY_Mosaic.tif"))
        
        for vhr_path in vhr_files:
            refid = vhr_path.stem.replace("_RGBY_Mosaic", "")
            mask_path = self.mask_dir / f"{refid}_mask.tif"
            
            if mask_path.exists():
                pairs.append((vhr_path, mask_path))
        
        return pairs
    
    def __len__(self) -> int:
        """Total number of patches per epoch."""
        return len(self.pairs) * self.patches_per_image
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and return a random patch from a VHR-mask pair.
        
        Args:
            idx: Index for the patch
        
        Returns:
            img_patch: torch.float32, shape (6, patch_size, patch_size)
            mask_patch: torch.int64, shape (patch_size, patch_size)
        """
        pair_idx = idx // self.patches_per_image
        vhr_path, mask_path = self.pairs[pair_idx]
        
        with rasterio.open(vhr_path) as src:
            img = src.read()
        
        with rasterio.open(mask_path) as src_m:
            mask = src_m.read(1)
        
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
        
        if self.mean is not None and self.std is not None:
            img_patch = (img_patch - self.mean) / (self.std + 1e-6)
        else:
            max_val = img_patch.max()
            if max_val > 0:
                img_patch = img_patch / max_val
        
        if self.augment:
            if random.random() > 0.5:
                img_patch = torch.flip(img_patch, dims=[2])
                mask_patch = torch.flip(mask_patch, dims=[1])
            
            if random.random() > 0.5:
                img_patch = torch.flip(img_patch, dims=[1])
                mask_patch = torch.flip(mask_patch, dims=[0])
        
        return img_patch, mask_patch


def estimate_mean_std(
    dataset: SimpleHablossPatchDataset,
    num_samples: int = 2000,
) -> Tuple[list, list]:
    """
    Estimate per-channel mean and std from a sample of patches.
    
    Args:
        dataset: SimpleHablossPatchDataset instance (should have mean/std=None)
        num_samples: Number of random patches to sample for statistics
    
    Returns:
        mean: List of 6 mean values (one per channel)
        std: List of 6 std values (one per channel)
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
