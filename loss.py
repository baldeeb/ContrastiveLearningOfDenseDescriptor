# pixelwise contrastive loss
import torch
import torch.nn.functional as F


def get_match_loss(descriptors, matches):
    assert(len(descriptors) == len(matches) == 2)
    matched_descriptors = [descriptors[i][:, matches[i][0], matches[i][1]] for i in range(2)]
    num_matches = len(matches[0])
    match_loss = 1.0 / num_matches * (matched_descriptors[0] - matched_descriptors[1]).pow(2).sum()
    return match_loss


# Computes the max(0, M - D(I_a,I_b,u_a,u_b))^2 term and
# normalizes using the number of hard negatives.
# TODO: go back and justify the hard negative condition.
def get_nonmatch_loss(d, nonmatch_pair, M_d=0.5, M_p=100, L2=True):
    p0, p1 = nonmatch_pair[0], nonmatch_pair[1]
    delta = (d[:, p0[0], p0[1]] - d[:, p1[0], p1[1]]).norm(2, dim=0)
    loss = torch.clamp(M_d - delta, min=0)  #.pow(2)
    N = len(torch.nonzero(loss, as_tuple=False)) 
    if N !=0:
        loss = loss.sum() / N  # divided by hard negative samples
    else:
        loss = loss.mean()

    return loss


# # This seems to push all descriptors to be equal
# # :param: d is the descriptor
# # :param: one_pixel is randomly selected background pixel locations (u, v) ??!!!
# # :param: mask is the background mask; all background pixels are 1 others are zero
# def one_pixel_with_masked_pixels(d, one_pixel, mask, M_d=0.0):
#     total_loss = 0
#     for i in range(len(d)):
#         if len(one_pixel[i]) > 0:
#             bg_d = d[i, :, mask[i]]
#             D = d[i, :, one_pixel[i][1], one_pixel[i][0]]
#             total_loss += torch.clamp(D.t().mm(bg_d)+M_d, min=0).mean()
#     return total_loss


# def divergence_loss_single_object(d, fg_mask):
#     loss = 0
#     N = 0
#     for i, masks in enumerate(fg_mask):
#         N += len(masks)
#         for obj in masks:
#             D = d[i, :, masks[obj][1], masks[obj][0]]
#             average_distance = (D - D.mean(dim=1, keepdim=True)).norm(dim=0).mean()
#             loss += torch.clamp(0.5-average_distance, min=0)
#     return loss/N


def encourage_unit_gaussian_distribution(d):
    """
    Intended to push the distribution of descriptors not to collapse or explode.
    Encourages the representation to be a unit hyper-sphere
    :param: d has shape NxCxHxW
    """
    mean = d.mean(dim=(2,3), keepdim=True)
    sigma_esc = (d - mean).norm(dim=1)
    return torch.clamp((0.5 - sigma_esc).mean(), min=0)

def get_loss(descriptors, matching_pairs, non_match_pair, w_match=1, w_bg=1, w_single=1):
    """
    Compares 2 descriptors expects a batch of 2
    :param: matching_pairs is a list of 2 tensors containing ordered indices of matching pairs of pixels
    """
    assert(descriptors.shape[0] == 2)
    loss = {}
    single_obj_threshold = 0.5

    d_a, d_b = descriptors[0], descriptors[1]
    descriptors = [d_a.sigmoid(), d_b.sigmoid()]
    
    if matching_pairs is not None:
        loss['match'] = get_match_loss(descriptors, torch.stack(matching_pairs)) * w_match
    
    loss['non_match'] = 0
    for i, (matches, non_matches, descriptor) in enumerate(zip(matching_pairs, non_match_pair, descriptors)): # removed torch.stack op to allow different n_pair/nonpair within batch
        for n in non_matches:
            loss['non_match'] += get_nonmatch_loss(descriptor, [matches, n], single_obj_threshold, 50) * w_single
    
    # loss['divergence'] = encourage_unit_gaussian_distribution(torch.stack(descriptors))       
    
    # return { 'match': match_loss, 
    #          'non_bg': non_match_loss_bg, 
    #          'non_single': non_match_loss_single, 
    #          'divergence_loss': divergence_loss}
    return loss


def neg_log_likelihood_loss(descriptors, pos_samples, neg_samples_list):
    '''
    an attempt at an NCE like loss
    '''
    assert(descriptors.shape[0] == 2)
    pos, neg = [], []
    for i in range(2):
        pos.push_back(descriptors[i, :, pos_samples[i][0], pos_samples[i][1]]) 
        neg.push_back([descriptors[i, :, n[0], n[1]] for n in neg_samples_list])

    print(pos[0].shape, neg[0][0].shape)

    # TODO:
    # - get dot between positives
    # - 
                
    