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

    normalize = False
    num_classes = 3 

# class temp():
#     pass

# log = temp()
# log.frequency.images = 
    

#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################

from tqdm import tqdm 
import torch
from dataset import make_data_loader, sample_from_augmented_pair
from loss import get_loss
from datetime import datetime
import os

from torch.utils.tensorboard import SummaryWriter
from util.model_storage import save_model

##TEMP#############
import matplotlib.pyplot as plt
import cv2
###################

from model import DenseModel


if __name__ == '__main__':
    device = torch.device(cfg.device)

    dataloader = make_data_loader(split='train', args=cfg())
    model = DenseModel(3, False, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    timestamp_str = datetime.now().now().strftime("%d_%m_%Y__%H_%M_%S")
    checkpoint_dir = f"results/checkpoints_{timestamp_str}/"
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        os.makedirs(f'{checkpoint_dir}/models/')
        os.makedirs(f'{checkpoint_dir}/images/')
    summary = SummaryWriter(log_dir=f"{checkpoint_dir}/runs")
    print(f"storing this run in: {checkpoint_dir}")

    loss_accumulated = []
    IMAGE_SHAPE = (480, 640)

    for epoch in range(50):
        for i, (images, batch) in enumerate(tqdm(dataloader)):

            images = images.to(device)
            descriptors = model(images)
            inv_desc0 = batch[0]['augmentor'].geometric_inverse(descriptors[0])
            inv_desc1 = batch[1]['augmentor'].geometric_inverse(descriptors[1])
            inv_descriptors = torch.stack((inv_desc0, inv_desc1))
            positive_samples, negative_samples = sample_from_augmented_pair(images, [batch[i]['augmentor'] for i in range(2)])
            if len(positive_samples) == 0 or len(negative_samples) == 0: 
                # TODO: catch in dataloader. 
                # print('HHHHHH got non overlapping masks')
                continue
            else:
                print('getting loss')
                loss = get_loss(inv_descriptors, positive_samples, negative_samples)
                loss_accumulated.append(torch.tensor(0.0).to(device))
                for k, v in loss.items(): 
                    summary.add_scalar(k, v)
                    loss_accumulated[-1] += v
                    if i % 20 == 0:
                        im = images[0].clone().detach()
                        im = batch[0]['augmentor'].de_normalize(im)
                        summary.add_image('image', torch.clamp(im0, min=0, max=1))
                        d = descriptors[0].clone().detach()
                        summary.add_image('descriptor',  torch.clamp(d.sigmoid(), min=0, max=1))
            if i % 1 == 0:
                (sum(loss_accumulated) / len(loss_accumulated)).backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_accumulated = []

            if i % 1000 == 0: 
                model_dir = f'{checkpoint_dir}/models/{timestamp_str}_{epoch}_{i}'
                save_model(model_dir, model, optimizer)

            # if i % 500 == 0:
            #     #################### Temp VISUALIZE ####################
            #     ########################################################
            #     fig = plt.figure()
            #     gs = fig.add_gridspec(1, 2)
            #     axs = gs.subplots()
            #     im = torch.clamp(images[0].clone().detach().cpu().permute(1,2,0).float(), min=0, max=1)
            #     d = torch.clamp(((descriptors[0]+1)/2).clone().detach().cpu().permute(1,2,0).float(), min=0, max=1)
            #     axs[0].imshow(im)
            #     axs[1].imshow(d)
            #     plt.savefig(f'{checkpoint_dir}/images/{epoch}_{i}')

    model_dir = f'{checkpoint_dir}/models/{timestamp_str}_{epoch}_final'
    save_model(model_dir, backbone, optimizer)













            # if False:  #i % 20 == 0:
            #     #################### Temp VISUALIZE ####################
            #     ########################################################
            #     fig = plt.figure()
            #     gs = fig.add_gridspec(1, 2)
            #     axs = gs.subplots()
            #     im = images[0].clone().detach().cpu().permute(1,2,0).float()
            #     d = ((descriptors[0]+1)/2).clone().detach().cpu().permute(1,2,0).float().numpy()
            #     axs[0].imshow(im)
            #     axs[1].imshow(d)
            #     plt.savefig(f'./results/{epoch}_{i}')

            # if False:
            #     # fig = plt.figure()
            #     # gs = fig.add_gridspec(2, 2)
            #     # axs = gs.subplots()
            #     # axs[0][0].imshow(images[0].clone().detach().cpu().permute(1,2,0)[1:].numpy())
            #     # axs[0][1].imshow(inv_images0.clone().detach().cpu().permute(1,2,0)[1:].numpy())
            #     # axs[1][0].imshow(images[1].clone().detach().cpu().permute(1,2,0)[1:].numpy())
            #     # axs[1][1].imshow(inv_images1.clone().detach().cpu().permute(1,2,0)[1:].numpy())

            #     plt.figure()
            #     plt.imshow(m.clone().detach().cpu().numpy())
            #     m2 = get_int32_mask_given_indices(s, IMAGE_SHAPE)
            #     m2[m2!=0] = 1
            #     plt.figure()
            #     plt.imshow(m2.clone().detach().cpu().numpy())
            #     m2 = get_int32_mask_given_indices(non, IMAGE_SHAPE)
            #     m2[m2!=0] = 1
            #     plt.figure()
            #     plt.imshow(m2.clone().detach().cpu().numpy())


            #     plt.show()

            # if False:  # TEMP: to control debug and vis
            #     #################### Temp DEBUG ####################
            #     ########################################################
            #     print('ims: ', images[0].shape, masks[0].shape)
            #     print('samples: ', samples[0].shape, non_matches[0].shape)
            #     print('descriptor: ', descriptors.shape)
            #     for t1, t2 in zip(*matched_indices):
            #         if samples_masks[0][t1[0], t1[1]] != samples_masks[1][t2[0], t2[1]]:
            #             print("ffffffuck")

            #     #################### Temp VISUALIZE ####################
            #     ########################################################
            #     fig = plt.figure()
            #     # fig.tight_layout()
            #     gs = fig.add_gridspec(2, 3)
            #     axs = gs.subplots()

            #     axs[0][0].imshow(images[0].permute(1,2,0))
            #     axs[0][1].imshow(samples_masks[0].detach().numpy())
            #     axs[0][2].imshow(masks[0].permute(1,2,0).detach().numpy())
            #     # axs[0][1].imshow(descriptors[0].permute(1,2,0)[1:].detach().numpy())
            #     # axs[0][2].scatter(samples[0][0], samples[0][1], alpha=0.3)

            #     axs[1][0].imshow(images[1].permute(1,2,0))
            #     axs[1][1].imshow(samples_masks[1].detach().numpy())
            #     axs[1][2].imshow(masks[1].permute(1,2,0).detach().numpy())
            #     # axs[1][1].imshow(descriptors[1].permute(1,2,0)[1:].detach().numpy())
            #     # axs[1][2].scatter(samples[1][0], samples[1][1], alpha=0.3)    
                
            #     plt.show()