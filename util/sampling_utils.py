import random
import torch
import numpy as np
from augmentations.util import union_of_augmented_images_in_original


def sample_from_mask(mask, k, blur=None):
    if blur is not None:
        assert(len(blur) == 2)
        # TODO: blur mask using a gaussian kernel
    mask_indices = (mask.squeeze() == 1).nonzero()
    k = min(k, mask_indices.shape[0])
    samples = random.sample(mask_indices.numpy().tolist(), k)
    return torch.tensor(samples).T

def sample_non_matches(sampled_indices, radius_mean, radius_sigma, limits=None):
    assert(torch.is_tensor(limits) or limits is None)
    if not isinstance(radius_mean, list): radius_mean = [radius_mean]
    if not isinstance(radius_sigma, list): radius_sigma = [radius_sigma]
    negatives = []
    num_samples = sampled_indices.shape[1]
    for m, s in zip(radius_mean, radius_sigma):
        angle_sampler = torch.distributions.uniform.Uniform(-np.pi, np.pi)
        thetas = angle_sampler.rsample([num_samples])
        radii = torch.normal(m, s, size=[num_samples])
        offsets = torch.stack((radii * torch.cos(thetas), radii * torch.sin(thetas)), dim=1).long()
        N = sampled_indices + offsets.T
        if limits is not None:
            limits = limits.long()
            N = N.where(N[:] > 0, torch.tensor(0).long())
            N[0] = N[0].where(N[0] < limits[0], limits[0]-1)
            N[1] = N[1].where(N[1] < limits[1], limits[1]-1)
        negatives.append(N.reshape((2, -1)))
    if len(negatives) == 1: return negatives[0]
    else: return negatives


def sample_from_augmented_pair(
    image_dim, augmentors, 
    num_samples,  
    neg_sample_mean_dist, 
    neg_sample_sigmas, 
    ROI_mask=None):
    '''
    images are of shape NxCxHxW
    augmentors is a list of N invertible functions used to augment.
    ROI_mask: a torch tensor containing 1 where sampling is accepted and zeros elsewhere
    '''
    mask = union_of_augmented_images_in_original(augmentors, image_dim)
    if ROI_mask is not None: mask *= ROI_mask
    samples = sample_from_mask(mask, num_samples)
    if len(samples) == 0: 
        return samples, samples
    else:
        non_matches = sample_non_matches(samples, radius_mean=neg_sample_mean_dist, 
            radius_sigma=neg_sample_sigmas,limits=torch.tensor(image_dim))
        return samples, non_matches


def get_samples(metas, 
    num_samples=2000,  
    neg_sample_mean_dist=[100, 150, 175], 
    neg_sample_sigmas=[10, 30, 50], 
    binary_mask=None):
    augmentors = [metas[i]['augmentor'] for i in range(2)]
    positive_samples, negative_samples = sample_from_augmented_pair(
                                            IMAGE_SHAPE, augmentors,
                                            num_samples=num_samples,    
                                            neg_sample_mean_dist=neg_sample_mean_dist,
                                            neg_sample_sigmas=neg_sample_sigmas,
                                            ROI_mask=binary_mask)
                                            