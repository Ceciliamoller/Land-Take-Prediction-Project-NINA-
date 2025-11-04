from pathlib import Path
import rasterio
import torch
from torch.utils.data import Dataset
from src.config import VHR_DIR, MASK_DIR

class HablossSampleDataset(Dataset):
    def __init__(self, ids):
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        fid = self.ids[idx]
        vhr_path = VHR_DIR / f"{fid}_RGBY_Mosaic.tif"
        mask_path = MASK_DIR / f"{fid}_mask.tif"

        with rasterio.open(vhr_path) as src:
            img = src.read() # (C, H, W)
        with rasterio.open(mask_path) as src_m:
            mask = src_m.read(1) # (H, W)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).long()

        return img, mask
