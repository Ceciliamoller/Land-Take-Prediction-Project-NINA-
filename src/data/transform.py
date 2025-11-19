import torch
import random
    
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