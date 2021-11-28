import torch
from augmentations.transforms import RandomResizedCrop, RandomHorizontalFlip
from  torchvision.transforms import Compose, RandomApply, ColorJitter, RandomGrayscale, GaussianBlur, Normalize, ToTensor 
from util import image_de_normalize

# data_mean, data_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet
data_mean, data_std = [0.5183, 0.5747, 0.7210], [0.3218, 0.3045, 0.2688]  # Unreal Progress Mugs

class GeometricallyInvertibleAugmentation():
    '''Derived from a re-implementation of SimCLR'''
    def __init__(self, image_size, s=1.0, img_stats=[data_mean, data_std]):
        assert(len(image_size) == 2)
        
        self.data_mean = torch.tensor(img_stats[0])
        self.data_std = torch.tensor(data_std[1])

        #############################################################
        # TODO: revisit
        gaussian_kernel_size = [i // 20 * 2 + 1 for i in image_size] 
        #############################################################
        
        self.random_resize_crop = RandomResizedCrop(image_size, scale=(0.2, 1.0))
        random_horizontal_flip = RandomHorizontalFlip()

        self.Normalize = Normalize(mean=self.data_mean, std=self.data_std)

        self.composed_transform = Compose([
            self.random_resize_crop,
            random_horizontal_flip,
            RandomApply(
                [ColorJitter(0.8*s,0.8*s,0.8*s,0.2*s)], 
                p=0.8),
            RandomGrayscale(p=0.2),
            RandomApply(
                [GaussianBlur(kernel_size=gaussian_kernel_size, sigma=(0.1, 2.0))], 
                p=0.5),
            self.Normalize
        ])

        self.inverse_random_resize_crop = self.random_resize_crop.get_inverse()
        self.geometric_inverse_transform = Compose([
            random_horizontal_flip.get_inverse(),
            self.inverse_random_resize_crop
        ])
        self.geometric_transform = Compose([
            random_horizontal_flip,
            self.random_resize_crop
        ])

    def __to_tensor(self, x):
        if not torch.is_tensor(x): x = ToTensor()(x)
        return x 

    def __call__(self, x):
        x = self.__to_tensor(x)
        # return self.transform(x)
        x = self.composed_transform(x)
        return x

    def geometric_inverse(self, x):
        x = self.__to_tensor(x)
        return self.geometric_inverse_transform(x)

    def geometric_only(self, x):
        x = self.__to_tensor(x)
        return self.geometric_transform(x)

    def _resize(self, shape):
        self.inverse_random_resize_crop.resize(shape)
        self.random_resize_crop.resize(shape)

    def de_normalize(self, x, device='cpu'):
        return image_de_normalize(x, 
            data_mean=self.data_mean, 
            data_std=self.data_std,
            device=device)

    # def __call__(self, x):
    #     x1 = self.transform(x)
    #     x2 = self.transform(x)
    #     return x1, x2 
