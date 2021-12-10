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



unreal_data_mean, unreal_data_std = [0.5183, 0.5747, 0.7210], [0.3218, 0.3045, 0.2688]  # Unreal Progress Mugs

def union_of_augmented_images_in_original(augmentors):
    '''
    returns a mask of a common region in the original image where those images overlap
    '''
    image_size = augmentors[0].image_size
    mask = torch.ones(image_size)
    for aug in augmentors:
        ones = torch.ones(tuple([1,*image_size]))
        mask_in_original_image = aug.geometric_inverse(ones)
        mask = mask * mask_in_original_image.squeeze()
    return mask

'''
Expects x shape: BxCxHxW
'''
def image_de_normalize(
    x, data_mean=torch.tensor(unreal_data_mean), 
    data_std=torch.tensor(unreal_data_std), 
    device=torch.device('cuda:0') ):
    m = data_mean.reshape(1, -1, 1, 1).to(device)
    s = data_std.reshape(1, -1, 1, 1).to(device)
    return (x * s) + m