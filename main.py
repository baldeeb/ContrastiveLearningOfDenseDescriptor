#temp config 
class cfg():
    dataset = 'unreal_parts'
    data_dir = '../simple_data'
    input_mode = 'RGB'
    obj_class = 'mug'
    n_pair = 0
    n_nonpair_singleobj = 0 
    n_nonpair_bg = 0
    batch_size = 1
    workers = 1

#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################

from tqdm import tqdm 
import torch
from torchvision.models.segmentation import deeplabv3_resnet50
from dataset import make_data_loader, sample_from_mask, sample_non_matches
from augmentations.geometrically_invertible_aug import GeometricallyInvertibleAugmentation as Augmentor
from loss import get_loss

##TEMP#############
from util.unsorted_utils import get_batch_masks, get_int32_mask_given_indices, get_matches_from_int_masks

import matplotlib.pyplot as plt
import cv2
###################

if __name__ == '__main__':
    backbone = deeplabv3_resnet50(num_classes=4)
    dataloader = make_data_loader(split='train', args=cfg())

    optimizer = torch.optim.Adam(backbone.parameters(), lr=0.001)

    for i, (images, batch) in enumerate(tqdm(dataloader)):

        masks = get_batch_masks(batch)
        samples, non_matches = [], []
        samples_masks = [] 
        for mask in masks: 
            samples.append(sample_from_mask(mask, 1000))
            non_matches.append(sample_non_matches(samples[-1], sigma=50))
            samples_masks.append(get_int32_mask_given_indices(samples[-1], (480, 640)))
            non_matches_mask.append(sample_non_matches(samples[-1], sigma=50))

        matched_indices = get_matches_from_int_masks(samples_masks[0], samples_masks[1])
        descriptors = backbone(images)['out']

        get_loss(descriptors, matched_indices, batch)  # TODO get matching pairs of pixels
        
        if i % 16 == 0:
            optimizer.step()
            optimizer.zero_grad()

        if True:
            fig = plt.figure()
            gs = fig.add_gridspec(1, 2)
            axs = gs.subplots()
            axs[0].imshow(images[0].permute(1,2,0))
            axs[1].imshow(descriptors[0].permute(1,2,0)[1:].detach().numpy())
            plt.savefig(f'./results/{i}')

        if False:  # TEMP: to control debug and vis

            #################### Temp DEBUG ####################
            ########################################################

            print('ims: ', images[0].shape, masks[0].shape)
            print('samples: ', samples[0].shape, non_matches[0].shape)
            print('descriptor: ', descriptors.shape)

            for t1, t2 in zip(*matched_indices):
                if samples_masks[0][t1[0], t1[1]] != samples_masks[1][t2[0], t2[1]]:
                    print("ffffffuck")

            #################### Temp VISUALIZE ####################
            ########################################################

            fig = plt.figure()
            # fig.tight_layout()
            gs = fig.add_gridspec(2, 3)
            axs = gs.subplots()

            axs[0][0].imshow(images[0].permute(1,2,0))
            axs[0][1].imshow(samples_masks[0].detach().numpy())
            axs[0][2].imshow(masks[0].permute(1,2,0).detach().numpy())
            # axs[0][1].imshow(descriptors[0].permute(1,2,0)[1:].detach().numpy())
            # axs[0][2].scatter(samples[0][0], samples[0][1], alpha=0.3)

            axs[1][0].imshow(images[1].permute(1,2,0))
            axs[1][1].imshow(samples_masks[1].detach().numpy())
            axs[1][2].imshow(masks[1].permute(1,2,0).detach().numpy())
            # axs[1][1].imshow(descriptors[1].permute(1,2,0)[1:].detach().numpy())
            # axs[1][2].scatter(samples[1][0], samples[1][1], alpha=0.3)    
            
            plt.show()
        
