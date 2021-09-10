# pixelwise contrastive loss
import torch
import torch.nn.functional as F

def knn(D1, D2):
    with torch.no_grad():
        dist = (D1.unsqueeze(2)-D2.unsqueeze(1)).pow(2).sum(0)
        min_dist, ind = dist.min(dim=0)

        return min_dist, ind

def img_to_world(img_index, depth, fx=320, fy=320, cx=320, cy=240):
    depth = depth.cpu()
    K_inv = torch.tensor([[1 / fx, 0, -cx / fx], [0, 1 / fy, -cy / fy], [0, 0, 1]], dtype=torch.float)
    xyz_normalized = torch.matmul(K_inv, torch.cat([img_index, torch.ones_like(img_index[0,None])]))
    xyz = torch.mul(xyz_normalized, depth)
    return xyz.T

def world_to_img(xyz, fx=320, fy=320, cx=320, cy=240):
    xyz = xyz / xyz[-1] # shape: 3, 3*n
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float)
    img_index = torch.matmul(K, xyz)
    return torch.round(img_index[:-1])



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
    loss = torch.clamp(M_d - delta, min=0).pow(2)
    N = len(torch.nonzero(loss, as_tuple=False)) 
    if N !=0:
        loss = loss.sum() / N  # divided by hard negative samples
    else:
        loss = loss.mean()

    return loss


# # :param: pairs -> shape: 4 x C x H x W
# # :param: M_p threshold distance
# def l2_pixel_loss(pair, M_p):
#     delta = (pair[:2] - pair[2:]).float().norm(2, dim=0)
#     l2_loss = (1.0 / M_p * torch.clamp(delta, max=M_p))
#     return l2_loss


# This seems to push all descriptors to be equal
# :param: d is the descriptor
# :param: one_pixel is randomly selected background pixel locations (u, v) ??!!!
# :param: mask is the background mask; all background pixels are 1 others are zero
def one_pixel_with_masked_pixels(d, one_pixel, mask, M_d=0.0):
    total_loss = 0
    for i in range(len(d)):
        if len(one_pixel[i]) > 0:
            bg_d = d[i, :, mask[i]]
            D = d[i, :, one_pixel[i][1], one_pixel[i][0]]
            total_loss += torch.clamp(D.t().mm(bg_d)+M_d, min=0).mean()
    return total_loss

def divergence_loss_single_object(d, fg_mask):
    loss = 0
    N = 0
    for i, masks in enumerate(fg_mask):
        N += len(masks)
        for obj in masks:
            D = d[i, :, masks[obj][1], masks[obj][0]]
            average_distance = (D - D.mean(dim=1, keepdim=True)).norm(dim=0).mean()
            loss += torch.clamp(0.5-average_distance, min=0)
    return loss/N

# pushes the last descriptor pixel to 1 over objects
def fg_bg_labels_loss(d, mask):
    return F.binary_cross_entropy_with_logits(d, 1-mask.float())


def get_loss(descriptors, matching_pairs, non_match_pair, w_match=1, w_bg=1, w_single=1):
    """
    Compares 2 descriptors expects a batch of 2
    :param: matching_pairs is a list of 2 tensors containing ordered indices of matching pairs of pixels
    """
    assert(descriptors.shape[0] == 2)
    bg_fg_threshold = 1.0
    single_obj_threshold = 0.5

    d_a, d_b = descriptors[0], descriptors[1]
    # descriptors = [d_a[:-1].sigmoid(), d_b[:-1].sigmoid()]
    descriptors = [d_a[:3].sigmoid(), d_b[:3].sigmoid()]
    
    match_loss = None
    if matching_pairs is not None:
        match_pair = torch.stack(matching_pairs)
        match_loss = get_match_loss(descriptors, match_pair) * w_match
    
    divergence_loss, non_match_loss_bg, non_match_loss_single = 0, 0, 0
    for i, (matches, non_matches, descriptor) in enumerate(zip(matching_pairs, non_match_pair, descriptors)): # removed torch.stack op to allow different n_pair/nonpair within batch
        non_match_loss_single += get_nonmatch_loss(descriptor, [matches, non_matches], single_obj_threshold, 50) * w_single
    #     non_match_loss_bg += one_pixel_with_masked_pixels(d, [t['nonpair_bg'][:2] for t in targets], [t['bg_mask'] for t in targets])*w_bg
    #     non_match_loss_single += get_nonmatch_loss(d, [t['nonpair_singleobj'] for t in targets], single_obj_threshold, 50) * w_single
    #     divergence_loss += divergence_loss_single_object(d, [t['fg_mask'+str(i)] for t in targets])
       
    label_loss = 0
    # d_a_labels, d_b_labels = d_a[-1], d_b[-1]
    # for i, l in enumerate([d_a_labels, d_b_labels]):
    #     label_loss += fg_bg_labels_loss(l, torch.stack([t['bg_mask'] for t in targets]))
    
    return { 'match': match_loss, 
             'non_bg': non_match_loss_bg, 
             'non_single': non_match_loss_single, 
             'label':label_loss,
             'divergence_loss': divergence_loss}
