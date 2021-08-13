"""
Re-implements some geometric transform to allow them and adds an inverse capability.
Moves any random activity into a separate function called randomize which is called
at initialization.
"""

from torchvision.transforms import InterpolationMode 
from torchvision.transforms import RandomHorizontalFlip as OriginalRandomHorizontalFlip
from torchvision.transforms import RandomResizedCrop as OriginalRandomResizedCrop
import torchvision.transforms.functional as F
import torch 
import math
from copy import deepcopy

class RandomResizedCrop(OriginalRandomResizedCrop):
    def __init__(self,  size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=InterpolationMode.BILINEAR):
        super().__init__(size, scale, ratio, interpolation)
        self.size
        self.scale = scale 
        self.ration = ratio
        self.randomize()
        self.apply_inverse = False
    
    @property
    def is_geometric(self):
        return True

    def randomize(self):
        """
        Equivalent to get_params in RandomResizeCrop, except it simply set the class paramters needed instead
        of returning them. 
        """
        height, width = self.size
        area = height * width

        log_ratio = torch.log(torch.tensor(self.ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            self.w = int(round(math.sqrt(target_area * aspect_ratio)))
            self.h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < self.w <= width and 0 < self.h <= height:
                self.i = torch.randint(0, height - self.h + 1, size=(1,)).item()
                self.j = torch.randint(0, width - self.w + 1, size=(1,)).item()
                return

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(self.ratio):
            self.w = width
            self.h = int(round(self.w / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            self.h = self.height
            self.w = int(round(self.h * max(self.ratio)))
        else:  # whole image
            self.w = width
            self.h = height
        self.i = (height - self.h) // 2
        self.j = (width - self.w) // 2
        return

    def forward(self, img):
        if self.apply_inverse:
            return self.inverse(img)
        return F.resized_crop(img, self.i, self.j, self.h, self.w, self.size, self.interpolation)

    def __get_discarded_side_lengths(self, ):
        top = int(self.i)
        bottom = int(self.size[0] - (self.h + self.i))
        left = int(self.j)
        right = int(self.size[1] - (self.w + self.j))
        return (left, top, right, bottom)    

    def inverse(self, img):
        padding = self.__get_discarded_side_lengths()
        result = F.resize(img, (self.h, self.w))
        result = F.pad(result, padding)
        return result

    def invert(self):
        self.apply_inverse = not self.apply_inverse

    def get_inverse(self):
        inv = deepcopy(self)
        inv.invert()
        return inv

class RandomHorizontalFlip(OriginalRandomHorizontalFlip):
    def __init__(self, p=0.5):
        super().__init__(p)
        self.randomize()
        
    @property
    def is_geometric(self):
        return True
    
    def randomize(self):
        self.flip = torch.rand(1) < self.p

    def forward(self, img):
        if self.flip: 
            return F.hflip(img)
        else:
            return img

    def inverse(self, img):
        return self.forward(img)

    def get_inverse(self):
        return self