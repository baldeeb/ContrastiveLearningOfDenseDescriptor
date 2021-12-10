''' Defines functions for performing pixelwise contrastive loss '''

import torch
from copy import deepcopy
import torch.nn.functional as F
from util.sampling_utils import get_samples 

def dot_pdt(source, samples):
    assert(len(source) == len(samples) == 2)
    pixel_pairs = torch.tensor([source[i][:, samples[i][0], samples[i][1]] for i in range(2)])
    return (pixel_pairs[0] * pixel_pairs[1]).sum(dim=0)

def L2_distances(source, samples):
    assert(len(source) == len(samples) == 2)
    pixel_pairs = [source[i][:, samples[i][0], samples[i][1]] for i in range(2)]
    return (pixel_pairs[0] - pixel_pairs[1]).norm(2, dim=0)


def get_match_loss(descriptors, matches):
    # sum_of_squared_distances = L2_distances(descriptors, matches).pow(2).sum()
    # TODO: are couple of lines below much different than using L2?
    matched_descriptors = [descriptors[i][:, matches[i][0], matches[i][1]] for i in range(2)]
    sum_of_squared_distances = (matched_descriptors[0] - matched_descriptors[1]).pow(2).sum()
 
    num_samples = len(matches[0])
    return sum_of_squared_distances / num_samples


# Computes the max(0, M - D(I_a,I_b,u_a,u_b))^2 term and
# normalizes using the number of hard negatives.
# TODO: go back and justify the hard negative condition.
def get_nonmatch_loss(d, nonmatch_pair, min_nonmatch_distance=0.5):
    distances = L2_distances([d, d], nonmatch_pair)
    loss = torch.clamp((min_nonmatch_distance - distances), min=0).pow(2)
    N = len(torch.nonzero(loss, as_tuple=False)) 
    if N !=0:
        loss = loss.sum() / N  # divided by hard negative samples
    else:
        loss = loss.mean()
    return loss


def encourage_unit_gaussian_distribution(d):
    """
    Intended to push the distribution of descriptors not to collapse or explode.
    Encourages the representation to be a unit hyper-sphere
    :param: d has shape NxCxHxW
    """
    mean = d.mean(dim=(2,3), keepdim=True)
    sigma_esc = (d - mean).norm(dim=1)
    return torch.clamp((0.5 - sigma_esc).mean(), min=0)

def contrastive_dense_loss(descriptors, positive_samples, negative_samples_list, w_match=1, w_non_matches=1):
    """
    Compares 2 descriptors expects a batch of 2
    :param: matching_pairs is a list of 2 tensors containing ordered indices of matching pairs of pixels
    """
    assert(descriptors.shape[0] == 2)
    assert(positive_samples is not None)
    assert(negative_samples_list is not None)
    loss = {}

    descriptors = [d.sigmoid() for d in descriptors]
    
    loss['match'] = get_match_loss(descriptors, torch.stack([positive_samples, positive_samples])) * w_match
    
    loss['non_match'] = 0
    for d in descriptors:
        for neg in negative_samples_list:
            loss['non_match'] += get_nonmatch_loss(d, [positive_samples, neg], 0.5) * w_non_matches
    
    # loss['divergence'] = encourage_unit_gaussian_distribution(torch.stack(descriptors))
    
    return loss

def contrastive_augmentation_loss(descriptors, metas, ROI_mask=None):
    inv_desc0 = metas[0]['augmentor'].geometric_inverse(descriptors[0])
    inv_desc1 = metas[1]['augmentor'].geometric_inverse(descriptors[1])
    inv_descriptors = torch.stack((inv_desc0, inv_desc1))
    positive_samples, negative_samples = get_samples(metas, binary_mask=ROI_mask)
    if len(positive_samples) == 0 or len(negative_samples) == 0: 
        return {}             
    return contrastive_dense_loss(inv_descriptors, positive_samples, negative_samples)
    

## NOTE: Not clear this benefits the goal
# def neg_log_likelihood_loss(descriptors, pos_samples, neg_samples_list):
#     '''
#     an attempt at an NCE like loss
#     '''
#     assert(descriptors.shape[0] == 2)
#     pos, neg = [], []
#     for i in range(2):
#         pos.push_back(descriptors[i, :, pos_samples[i][0], pos_samples[i][1]]) 
#         neg.push_back([descriptors[i, :, n[0], n[1]] for n in neg_samples_list])

#     print(pos[0].shape, neg[0][0].shape)

#     # TODO:
#     # - get dot between positives
#     # - 
                
    


def get_descriptor_scale(descriptor_pair, metas):
    dw, dh = tensor

def pyramidal_contrastive_augmentation_loss(descriptors_pyramid, metas):
    '''
    This loss uses different scales which could be produced by a pyramidal network 
    to derive a contrastive loss
    

    descriptors: list of pairs of dense descriptors. First pair has largest 
        Each pair is of the same scale.
        Each pair is derived by augmenting the same image.  

    metas: include the augmentations used on the original image.
    '''

    # Decide how to sample in original image
    num_positive_samples=1000,  
    neg_sample_mean_dist=[100, 150, 175], 
    neg_sample_sigmas=[10, 30, 50], 

    # TODO: get scale of images relative to each other from descriptors
    image_scales = [1, 0.5, 0.25]

    original_shape = torch.tensor(metas['images'].shape[-2:])

    loss_dicts = []
    for descriptor_pair in descriptors_pyramid:

        descriptor_shape = descriptor_pair[0].shape[-2:]
        scale = descriptor_shape / original_shape 
        
        neg_means = neg_sample_mean_dist * scale
        neg_std = neg_sample_sigmas * scale
        num_samples = num_positive_samples * scale

        aug_copies = [deepcopy(metas[i]['augmentor']) for i in range(2)]
        augmentors = [aug.re_shape(descriptor_shape) for aug in aug_copies]

        # Get samples
        positive_samples, negative_samples = sample_from_augmented_pair(
                                        augmentors,
                                        num_samples=num_samples,    
                                        neg_sample_mean_dist=neg_sample_mean_dist,
                                        neg_sample_sigmas=neg_sample_sigmas,
                                        ROI_mask=binary_mask)
        if len(positive_samples) == 0 or len(negative_samples) == 0: 
            continue        

        # Invert Geometric augmentations
        re_adjusted_descriptors = []
        for aug, d in zip(augmentors, descriptor_pair):
            re_adjusted_descriptors.append(augmentors.geometric_inverse(d))
        re_adjusted_descriptors = torch.stack(tuple(re_adjusted_descriptors))

        loss_dicts.append(
            contrastive_dense_loss(
                re_adjusted_descriptors, 
                positive_samples, 
                negative_samples))
    
    # merge loss dicts
    loss = {k: [ld[k] for ld in loss_dicts] for k in loss_dicts[0]}
    return loss 
