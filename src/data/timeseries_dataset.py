from pathlib import Path
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from src.config import (
    SENTINEL_DIR,
    VHR_DIR,
    MASK_DIR,
)


def find_file_by_prefix(base_dir: Path, fid: str) -> Path:
    """
    Find a file in base_dir whose name starts with fid and ends with .tif or .tiff.

    Example:
        fid = "a22-0323..._37-98..."
        file = "a22-0323..._37-98..._RGBNIRRSWIRQ_Mosaic.tif"

    This assumes there is exactly one such file per fid.
    """
    candidates = sorted(
        list(base_dir.glob(f"{fid}*.tif")) +
        list(base_dir.glob(f"{fid}*.tiff"))
    )
    if not candidates:
        raise FileNotFoundError(f"No file starting with {fid} in {base_dir}")
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple files starting with {fid} in {base_dir}: {candidates}")
    return candidates[0]


class TimeSeriesDataset(Dataset):
    """
    Loads time series data and reshapes it into (T, C, H, W)
    so it can be fed directly to temporal models (like the FCEF baseline).

    Works with pre-cropped 64×64 chips. Each chip is a single sample.

    Assumptions:
      - `ids` are REFIDs that match the *prefix* of the filenames in
        SENTINEL_DIR / VHR_DIR / MASK_DIR.
      - All chips are already cropped to 64×64 pixels on disk.
    """

    def __init__(
        self,
        ids,
        transform,
        sensor: str = "sentinel",
        slice_mode: str = None,
    ):
        """
        ids: list of REFIDs (filename stems without the long suffix)
        sensor: "sentinel" or "vhr"
        slice_mode: None or "first_half"
        transform: Transform to apply (flips, rotations, normalization)
        """
        self.ids = ids
        self.sensor = sensor.lower()
        self.slice_mode = slice_mode
        self.transform = transform

        # Pre-resolve image and mask paths once for stability and speed
        self.img_paths: dict[str, Path] = {}
        self.mask_paths: dict[str, Path] = {}

        for fid in self.ids:
            if self.sensor == "sentinel":
                img_path = find_file_by_prefix(SENTINEL_DIR, fid)
            elif self.sensor == "vhr":
                img_path = find_file_by_prefix(VHR_DIR, fid)
            else:
                raise ValueError(f"Unknown sensor: {self.sensor}")

            mask_path = find_file_by_prefix(MASK_DIR, fid)

            self.img_paths[fid] = img_path
            self.mask_paths[fid] = mask_path

    def __len__(self):
        # One sample per chip
        return len(self.ids)

    def __getitem__(self, idx):
        # Direct mapping: each idx corresponds to one 64×64 chip
        fid = self.ids[idx]

        img_path = self.img_paths[fid]
        mask_path = self.mask_paths[fid]

        # 1) read arrays
        with rasterio.open(img_path) as src:
            img = src.read()  # (bands, H, W)
        with rasterio.open(mask_path) as src_m:
            mask = src_m.read(1)  # (H, W)

        # 2) reshape to (T, C, H, W) depending on sensor
        if self.sensor == "sentinel":
            # Expected layout: 126 = 7 years * 2 quarters * 9 bands
            num_bands, H, W = img.shape
            if num_bands != 126:
                raise ValueError(
                    f"Expected 126 bands for Sentinel, got {num_bands} for {fid} at {img_path}"
                )
            img = img.reshape(7, 2, 9, H, W)
            img = img.reshape(14, 9, H, W)

        elif self.sensor == "vhr":
            # Expected layout: 6 = 2 times * 3 bands
            num_bands, H, W = img.shape
            if num_bands != 6:
                raise ValueError(
                    f"Expected 6 bands for VHR, got {num_bands} for {fid} at {img_path}"
                )
            img = img.reshape(2, 3, H, W)

        # 3) optionally take first half of the time series
        if self.slice_mode == "first_half":
            T = img.shape[0]
            img = img[: T // 2]

        # 4) to torch tensors
        img = torch.from_numpy(img).float()     # (T, C, H, W)
        mask = torch.from_numpy(mask).long()    # (H, W)
        mask = (mask > 0).long()
        
        # 5) Apply transforms (which handle padding/cropping via CenterCropTS)
        if self.transform is not None:
            img, mask = self.transform(img, mask)

        return img, mask
