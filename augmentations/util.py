import torch
    
# def union_of_augmented_images_in_original(images, augmentors):
#     '''
#     returns a mask of a common region in the original image where those images overlap
#     '''
#     sh = images.shape
#     masks = torch.zeros((sh[0], sh[2], sh[3]))
#     for i in range(sh[0]):
#         inv = augmentors[i].geometric_inverse(images[i])
#         collapsed = inv.sum(dim=0).squeeze()
#         masks[i, collapsed!=0] = 1
#     return torch.prod(masks, 0)


def union_of_augmented_images_in_original(augmentors, image_dim):
    '''
    returns a mask of a common region in the original image where those images overlap
    '''
    assert(isinstance(image_dim, tuple) and len(image_dim) == 2)
    mask = torch.ones(image_dim)
    for aug in augmentors:
        ones = torch.ones(tuple([1,*image_dim]))
        mask_in_original_image = aug.geometric_inverse(ones)
        mask = mask * mask_in_original_image.squeeze()
    return mask

