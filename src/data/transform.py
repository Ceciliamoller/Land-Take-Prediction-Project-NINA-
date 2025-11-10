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