#temp config 
class cfg():
    dataset = 'unreal_parts'
    data_dir = '../simple_data'
    image_type = 'RGB'
    obj_class = 'mug'
    n_pair = 0
    n_nonpair_singleobj = 0 
    n_nonpair_bg = 0
    batch_size = 1
    workers = 1

    device = 'cuda:0'
    # device = 'cpu'

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

import torch.multiprocessing as mp

##TEMP#############
from util.unsorted_utils import get_batch_masks, get_int32_mask_given_indices, get_matches_from_int_masks

import matplotlib.pyplot as plt
import cv2
###################

def sample_from_pair(images, augmentors):
    '''
    images are of shape NxCxHxW
    augmentors is a list of N invertible functions used to augment.
    '''
    sh = images.shape
    masks = torch.zeros((sh[0], sh[2], sh[3]))
    for i in range(2):
        inv = augmentors[i].geometric_inverse(images[i])
        collapsed = inv.sum(dim=0).squeeze()
        masks[i, collapsed!=0] = 1
    mask = torch.prod(masks, 0)
    samples = sample_from_mask(mask, 5000)
    
    if len(samples) == 0: return mask, samples, samples
    
    non_matches = sample_non_matches(samples, sigma=50, limits=(480, 640))
    return mask, samples, non_matches


if __name__ == '__main__':
    device = torch.device(cfg.device)

    dataloader = make_data_loader(split='train', args=cfg())
    backbone = deeplabv3_resnet50(num_classes=3).to(device)
    optimizer = torch.optim.Adam(backbone.parameters(), lr=0.0001)

    loss_accumulated = []
    IMAGE_SHAPE = (480, 640)

    for epoch in range(20):
        for i, (images, batch) in enumerate(tqdm(dataloader)):


            images = images.to(device)
            descriptors = backbone(images)['out']
            

            inv_desc0 = batch[0]['augmentor'].geometric_inverse(descriptors[0])
            inv_desc1 = batch[1]['augmentor'].geometric_inverse(descriptors[1])
            inv_descriptors = torch.stack((inv_desc0, inv_desc1))

            m, s, non = sample_from_pair(images, [batch[i]['augmentor'] for i in range(2)])
            if len(s) == 0 or len(non) == 0: 
                print('HHHHHH got non overlapping masks')
                continue
            else:
                matched_indices = [s, s]
                non_matching_indices = [non, non]
                loss = get_loss(inv_descriptors, matched_indices, non_matching_indices, w_match=0.5)
                loss_accumulated.append(torch.tensor(0.0).to(device))
                for l in loss.values(): loss_accumulated[-1] += l
                print(loss_accumulated[-1])
    
            if i % 1 == 0:
                (sum(loss_accumulated) / len(loss_accumulated)).backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_accumulated = []




            if i % 20 == 0:
                #################### Temp VISUALIZE ####################
                ########################################################
                fig = plt.figure()
                gs = fig.add_gridspec(1, 2)
                axs = gs.subplots()
                im = images[0].clone().detach().cpu().permute(1,2,0).float()
                d = descriptors[0].clone().detach().cpu().permute(1,2,0)[1:].float().numpy()
                axs[0].imshow(im)
                axs[1].imshow(d)
                plt.savefig(f'./results/{epoch}_{i}')

            if False:
                # fig = plt.figure()
                # gs = fig.add_gridspec(2, 2)
                # axs = gs.subplots()
                # axs[0][0].imshow(images[0].clone().detach().cpu().permute(1,2,0)[1:].numpy())
                # axs[0][1].imshow(inv_images0.clone().detach().cpu().permute(1,2,0)[1:].numpy())
                # axs[1][0].imshow(images[1].clone().detach().cpu().permute(1,2,0)[1:].numpy())
                # axs[1][1].imshow(inv_images1.clone().detach().cpu().permute(1,2,0)[1:].numpy())

                plt.figure()
                plt.imshow(m.clone().detach().cpu().numpy())
                m2 = get_int32_mask_given_indices(s, IMAGE_SHAPE)
                m2[m2!=0] = 1
                plt.figure()
                plt.imshow(m2.clone().detach().cpu().numpy())
                m2 = get_int32_mask_given_indices(non, IMAGE_SHAPE)
                m2[m2!=0] = 1
                plt.figure()
                plt.imshow(m2.clone().detach().cpu().numpy())


                plt.show()

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
        
