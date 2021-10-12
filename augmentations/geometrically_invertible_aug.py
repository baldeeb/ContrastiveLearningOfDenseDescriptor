import torch
from augmentations.transforms import RandomResizedCrop, RandomHorizontalFlip
from  torchvision.transforms import Compose, RandomApply, ColorJitter, RandomGrayscale, GaussianBlur, Normalize, ToTensor 

# data_mean, data_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet
data_mean, data_std = [0.5183, 0.5747, 0.7210], [0.3218, 0.3045, 0.2688]  # Unreal Progress Mugs

class GeometricallyInvertibleAugmentation():
    '''Derived from a re-implementation of SimCLR'''
    def __init__(self, image_size, s=1.0):
        assert(len(image_size) == 2)
        
        #############################################################
        # TODO: revisit
        gaussian_kernel_size = [i // 20 * 2 + 1 for i in image_size] 
        #############################################################
        
        random_resize_crop = RandomResizedCrop(image_size, scale=(0.2, 1.0))
        random_horizontal_flip = RandomHorizontalFlip()

        self.composed_transform = Compose([
            random_resize_crop,
            random_horizontal_flip,
            RandomApply(
                [ColorJitter(0.8*s,0.8*s,0.8*s,0.2*s)], 
                p=0.8),
            RandomGrayscale(p=0.2),
            RandomApply(
                [GaussianBlur(kernel_size=gaussian_kernel_size, sigma=(0.1, 2.0))], 
                p=0.5),
            Normalize(mean=data_mean, std=data_std)
        ])

        self.geometric_inverse_transform = Compose([
            random_horizontal_flip.get_inverse(),
            random_resize_crop.get_inverse()
        ])
        self.geometric_transform = Compose([
            random_horizontal_flip,
            random_resize_crop
        ])

    def __call__(self, x):
        if not torch.is_tensor(x):
            x = ToTensor()(x) 
        # return self.transform(x)
        x = self.composed_transform(x)
        return x

    def geometric_inverse(self, x):
        if not torch.is_tensor(x):
            x = ToTensor()(x)
        return self.geometric_inverse_transform(x)

    def geometric_only(self, x):
        if not torch.is_tensor(x):
            x = ToTensor()(x)
        return self.geometric_transform(x)

    def de_normalize(self, x):
        return (x / data_std) + data_mean
    # def __call__(self, x):
    #     x1 = self.transform(x)
    #     x2 = self.transform(x)
    #     return x1, x2 
