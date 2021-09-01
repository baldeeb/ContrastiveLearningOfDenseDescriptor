import torch
from augmentations.transforms import RandomResizedCrop, RandomHorizontalFlip
import torchvision.transforms as T

imagenet_mean_std = [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]

class GeometricallyInvertibleAugmentation():
    '''Derived from a re-implementation of SimCLR'''
    def __init__(self, image_size, mean_std=imagenet_mean_std, s=1.0):
        assert(len(image_size) == 2)
        gaussian_kernel_size = [i // 20 * 2 + 1 for i in image_size]
        random_resize_crop = RandomResizedCrop(image_size, scale=(0.2, 1.0))
        random_horizontal_flip = RandomHorizontalFlip()

        self.transform = T.Compose([
            random_resize_crop,
            random_horizontal_flip,
            T.RandomApply(
                [T.ColorJitter(0.8*s,0.8*s,0.8*s,0.2*s)], 
                p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply(
                [T.GaussianBlur(kernel_size=gaussian_kernel_size, sigma=(0.1, 2.0))], 
                p=0.5),
            T.Normalize(*mean_std)
        ])

        self.geometric_inverse_transform = T.Compose([
            random_horizontal_flip.get_inverse(),
            random_resize_crop.get_inverse()
        ])
        self.geometric_transform = T.Compose([
            random_horizontal_flip,
            random_resize_crop
        ])

    def __call__(self, x):
        if not torch.is_tensor(x):
            x = T.ToTensor()(x) 
        return self.transform(x)

    def geometric_inverse(self, x):
        if not torch.is_tensor(x):
            x = T.ToTensor()(x)
        return self.geometric_inverse_transform(x)

    def geometric_only(self, x):
        if not torch.is_tensor(x):
            x = T.ToTensor()(x)
        return self.geometric_transform(x)

    # def __call__(self, x):
    #     x1 = self.transform(x)
    #     x2 = self.transform(x)
    #     return x1, x2 
