import torch 

# TODO: move to dataset or related class and call get_binary_mask
# TODO: add paramaeter to specify what objects which masks we would want, otherwise get all.
def get_batch_masks(batch_data):
    ''' returns batch of binary masks of all objects. '''
    masks = []
    for data in batch_data:
        mask = torch.zeros_like(data['mask'])
        for obj_indices in data['objmaskid'].values():
            for i in obj_indices:
                mask = torch.logical_or(mask, (data['mask'] == i))
        masks.append(mask)
    return torch.stack((masks))


def get_int32_mask_given_indices(samples, shape):
    '''
    samples shape: 2 x N
    image shape: H x W
    NOTE: this will discard samples that are out of bound.
    '''
    samples = samples[:, samples[0, :] < shape[0]]
    samples = samples[:, samples[1, :] < shape[1]]
    mask = torch.zeros(shape, dtype=torch.int32)
    mask[samples[0,:], samples[1,:]] = torch.arange(1, samples.shape[1] + 1, 1, dtype=torch.int32)
    return mask


def get_sorted_list_of_nonzero_indices_and_their_values(m):
    '''
    input shape: 
    '''
    indices = (m != 0).nonzero().T
    values = m[indices[0], indices[1]]
    sorted_values, i = torch.sort(values)
    sorted_indices = indices[:, i]
    return sorted_indices.T, sorted_values.T

def get_matches_from_int_masks(m1, m2):
    '''
    given 2 masks with integer values, returns lists of indices that which have the same integer in mask.
    
    mask shape: 
    '''
    i1, v1 = get_sorted_list_of_nonzero_indices_and_their_values(m1) 
    i2, v2 = get_sorted_list_of_nonzero_indices_and_their_values(m2) 
    v1_expanded = v1.expand((v2.shape[0], *v1.shape)).T
    diff = (v1_expanded  - v2)
    matching_indices = (diff == 0).nonzero()
    if matching_indices.shape[0] == 0: return torch.tensor([]), torch.tensor([])
    v1_indices, v2_indices = [matching_indices[:, i].flatten() for i in range(2)]
    return [i1[v1_indices], i2[v2_indices]]
