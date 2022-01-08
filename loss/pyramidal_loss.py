from loss.contrastive_loss import contrastive_dense_loss
from util.sampling_utils import sample_from_augmented_pair
import numpy as np
from copy import deepcopy
import torch

def pyramidal_contrastive_augmentation_loss(
    descriptors_pyramid, 
    meta,
    num_positive_samples = 1000, 
    neg_sample_mean_dist = np.array([100, 150, 175]), 
    neg_sample_sigmas = np.array([10, 30, 50])):
    '''
    This loss uses different scales which could be produced by a pyramidal network 
    to derive a contrastive loss

    descriptors: list of pairs of dense descriptors. First pair has largest 
        Each pair is of the same scale.
        Each pair is derived by augmenting the same image.  

    meta: include the augmentations used on the original image.
    '''

    original_H, original_W = meta['image'].shape[-2:]

    loss_dicts = []
    for descriptor_pair in descriptors_pyramid:

        # Scale images and augmentations
        dH, dW = descriptor_pair[0].shape[-2:]
        scale = min(dH/original_H, dW/original_W) 
        
        neg_means = [m * scale for m in neg_sample_mean_dist]
        neg_std = [s * scale for s in neg_sample_sigmas]
        num_samples = int(num_positive_samples * scale)

        augmentors = [deepcopy(meta['augmentor']) for i in range(2)]
        [aug.re_shape((dH, dW)) for aug in augmentors]

        # Get samples
        positive_samples, negative_samples = sample_from_augmented_pair(
                                        augmentors,
                                        num_samples=num_samples,    
                                        neg_sample_mean_dist=neg_means,
                                        neg_sample_sigmas=neg_std)

        if len(positive_samples) == 0 or len(negative_samples) == 0: 
            continue        

        # Invert Geometric augmentations
        re_adjusted_descriptors = []
        for aug, d in zip(augmentors, descriptor_pair):
            re_adjusted_descriptors.append(aug.geometric_inverse(d))
        re_adjusted_descriptors = torch.stack(tuple(re_adjusted_descriptors))

        # Calculate loss
        loss_dicts.append(
            contrastive_dense_loss(
                re_adjusted_descriptors, 
                positive_samples, 
                negative_samples, 
                w_non_matches=0.5))

    # Merge loss dicts
    loss = {k: [ld[k] for ld in loss_dicts] for k in loss_dicts[0]}
    return loss 


def non_contrastive_loss(p, z):
    z = z.detach()
    
    p = torch.nn.functional.normalize(p)
    z = torch.nn.functional.normalize(z)

    return (p - z).norm(p=2, dim=0).mean()