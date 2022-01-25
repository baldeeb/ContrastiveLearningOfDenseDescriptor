import torch
from loss.contrastive_loss import contrastive_dense_loss
from util.sampling_utils import get_samples 

def contrastive_augmentation_loss(descriptors, meta, ROI_mask=None):
    inv_desc0 = meta[0]['augmentor'].geometric_inverse(descriptors[0])
    inv_desc1 = meta[1]['augmentor'].geometric_inverse(descriptors[1])
    inv_descriptors = torch.stack((inv_desc0, inv_desc1))
    positive_samples, negative_samples = get_samples(meta, binary_mask=ROI_mask)
    if len(positive_samples) == 0 or len(negative_samples) == 0: 
        return {}             
    return contrastive_dense_loss(inv_descriptors, positive_samples, negative_samples)
    

from augmentations.util import union_of_augmented_images_in_original
from loss.contrastive_loss import get_match_loss

def overlapping_region_positive_sample_loss(descriptors, meta):
    augmentors = [meta[i]['augmentor'] for i in range(2)]
    mask = union_of_augmented_images_in_original(augmentors)
    indices = (mask.squeeze() == 1).nonzero()
    return get_match_loss(descriptors, torch.stack([indices.T, indices.T]))