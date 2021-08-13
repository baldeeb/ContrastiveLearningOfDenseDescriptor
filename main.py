'''
Intended to lay down the high level scheme of training and data flow pertaining to that.
'''



# Setup network, optimizer, logger... the usual stuff


# Load batch of data and create a similarly sized batch of augmentation objects.
# augment data -> run through network -> apply geometrically inverted augmentation
#   -> select a set of non-zero pixels as positive pairs -> select negative pairs

# Initially: 
#   - batch_size = 1
#   - select most positive and negative pairs on object mask.
#

# TODO: 
#   - What information is needed?   ->    image, depth, 
#       
#

from torchvision.models.segmentation import deeplabv3_resnet50
from augmentations.geometrically_invertible_aug import GeometricallyInvertibleAugmentation as Augmentor
backbone = deeplabv3_resnet50()


for data in dataloader:

    aug = Augmentor()

    augmented = aug(data)
    descriptor = backbone(data)
    aug_descriptor = backbone(augmented)
    
    