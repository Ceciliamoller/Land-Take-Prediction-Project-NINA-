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

    Supports multiple patches per tile per epoch for fair comparison with U-Net.

    Assumptions:
      - `ids` are REFIDs that match the *prefix* of the filenames in
        SENTINEL_DIR / VHR_DIR / MASK_DIR.
    """

    def __init__(
        self,
        ids,
        transform,
        sensor: str = "sentinel",
        slice_mode: str = None,
        patches_per_image: int = 20,
    ):
        """
        ids: list of REFIDs (filename stems without the long suffix)
        sensor: "sentinel" or "vhr"
        slice_mode: None or "first_half"
        patches_per_image: Number of patches to sample per tile per epoch (default 20)
        """
        self.ids = ids
        self.sensor = sensor.lower()
        self.slice_mode = slice_mode
        self.transform = transform
        self.patches_per_image = patches_per_image

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
        # Total patches per epoch = num_tiles * patches_per_image
        return len(self.ids) * self.patches_per_image

    def __getitem__(self, idx):
        # Map global idx to tile idx (multiple patches per tile)
        tile_idx = idx // self.patches_per_image
        fid = self.ids[tile_idx]

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
            H, W = img.shape[1], img.shape[2]
            img = img.reshape(7, 2, 9, H, W)
            img = img.reshape(14, 9, H, W)

        elif self.sensor == "vhr":
            # Expected layout: 6 = 2 times * 3 bands
            H, W = img.shape[1], img.shape[2]
            img = img.reshape(2, 3, H, W)

        # 3) optionally take first half of the time series
        if self.slice_mode == "first_half":
            T = img.shape[0]
            img = img[: T // 2]

        # 4) to torch tensors
        img = torch.from_numpy(img).float()     # (T, C, H, W)
        mask = torch.from_numpy(mask).long()    # (H, W)
        mask = (mask > 0).long()

        if self.transform is not None:
            img, mask = self.transform(img, mask)

        return img, mask
