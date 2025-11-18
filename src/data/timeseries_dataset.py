from pathlib import Path
import rasterio
import torch
from torch.utils.data import Dataset

from src.config import (
    SENTINEL_DIR,
    PLANETSCOPE_DIR,
    VHR_DIR,
    MASK_DIR,
)

class TimeSeriesDataset(Dataset):
    """
    Loads ONE sensor per sample and reshapes it into (T, C, H, W)
    so it can be fed directly to the torchrs FD-CD models.
    """

    def __init__(self, ids, transform, sensor: str = "sentinel", slice_mode: str = None):
        """
        ids: list of REFIDs
        sensor: "sentinel", "planetscope", "vhr"
        slice_mode: None | "first_half"
        """
        self.ids = ids
        self.sensor = sensor.lower()
        self.slice_mode = slice_mode
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        fid = self.ids[idx]

        # 1) pick image path by sensor
        if self.sensor == "sentinel":
            img_path = SENTINEL_DIR / f"{fid}"
        elif self.sensor == "planetscope":
            img_path = PLANETSCOPE_DIR / f"{fid}"
        elif self.sensor == "vhr":
            img_path = VHR_DIR / f"{fid}"
        else:
            raise ValueError(f"Unknown sensor: {self.sensor}")

        mask_path = MASK_DIR / f"{fid}"

        # 2) read arrays
        with rasterio.open(img_path) as src:
            img = src.read()  # (bands, H, W)
        with rasterio.open(mask_path) as src_m:
            mask = src_m.read(1)  # (H, W)

        # 3) reshape to (T, C, H, W) depending on sensor
        if self.sensor == "sentinel":
            # 126 = 7 years * 2 quarters * 9 bands
            # img: (126, H, W) -> (7, 2, 9, H, W) -> (14, 9, H, W)
            H, W = img.shape[1], img.shape[2]
            img = img.reshape(7, 2, 9, H, W)
            img = img.reshape(14, 9, H, W)

        elif self.sensor == "planetscope":
            # 42 = 7 * 2 * 3
            H, W = img.shape[1], img.shape[2]
            img = img.reshape(7, 2, 3, H, W)
            img = img.reshape(14, 3, H, W)

        elif self.sensor == "vhr":
            # 6 = 2 * 3
            H, W = img.shape[1], img.shape[2]
            img = img.reshape(2, 3, H, W)

        # 4) optionally take first half of the time series
        if self.slice_mode == "first_half":
            T = img.shape[0]
            img = img[: T // 2]

        # 5) to torch
        img = torch.from_numpy(img).float()     # (T, C, H, W)
        mask = torch.from_numpy(mask).long()    # (H, W)
        mask = (mask > 0).long()

        if self.transform is not None:
            img, mask = self.transform(img, mask)

        return img, mask
