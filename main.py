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
    

class Logger():

    def __init__(self, model_save_rate=1000, visuals_save_rate = 20, loss_summary_rate=1, image_de_normalizer=None):
        self.model_save_rate = model_save_rate
        self.loss_summary_rate = loss_summary_rate
        self.visuals_save_rate = visuals_save_rate
        self.timestamp_str = datetime.now().now().strftime("%d_%m_%Y__%H_%M_%S")
        self.checkpoint_dir = f"results/checkpoints_{self.timestamp_str}/"
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            os.makedirs(f'{self.checkpoint_dir}/models/')
            os.makedirs(f'{self.checkpoint_dir}/images/')
        self.summary = SummaryWriter(log_dir=f"{self.checkpoint_dir}/runs")
        self.image_de_normalizer = image_de_normalizer
        self.counter = 0

    def update(self, epoch, model, optimizer, loss_dict, images, descriptor):
        self.epoch = epoch
        self.counter += 1
        if self.counter % self.model_save_rate == 0:
            self.save_model(model, optimizer)
        if self.counter % self.loss_summary_rate == 0:
            self.update_loss(loss_dict)
        if self.counter % self.visuals_save_rate == 0:
            self.save_visuals(images, descriptors)

    def save_visuals(self, images, descriptors, de_normalizer=None):
        im = images.clone().detach()
        if self.image_de_normalizer:
            im = self.image_de_normalizer(im)
        im = torch.clamp(im[0], min=0, max=1)
        self.summary.add_image('image', im)
        d = descriptors[0].clone().detach()
        d = torch.clamp(d.sigmoid(), min=0, max=1)
        self.summary.add_image('descriptor', d)

    def update_loss(self, loss):
        for k, v in loss.items(): 
            self.summary.add_scalar(k, v)

    def save_model(self, model, optimizer, model_dir_end=''):
        model_dir = f'{self.checkpoint_dir}/models/{self.timestamp_str}_{self.epoch}_{self.counter}{model_dir_end}'
        save_model(model_dir, model, optimizer)


#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################

import os
from tqdm import tqdm 
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import torch
from torchvision.transforms import Resize

from loss import get_loss
from model import DenseModel
from util.model_storage import save_model
from augmentations.util import image_de_normalize
from dataset import make_data_loader, sample_from_augmented_pair

##TEMP#############
import matplotlib.pyplot as plt
import cv2
###################


if __name__ == '__main__':
    device = torch.device(cfg.device)

    dataloader = make_data_loader(split='train', args=cfg())
    model = DenseModel(3, False, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    logger = Logger(image_de_normalizer=image_de_normalize)
    print(f"storing this run in: {logger.checkpoint_dir}")


    loss_accumulated = []
    IMAGE_SHAPE = (480, 640)

    for epoch in range(50):
        for sample_index, (images, batch) in enumerate(tqdm(dataloader)):

            images = images.to(device)
            descriptors = model(images)
            inv_desc0 = batch[0]['augmentor'].geometric_inverse(descriptors[0])
            inv_desc1 = batch[1]['augmentor'].geometric_inverse(descriptors[1])
            inv_descriptors = torch.stack((inv_desc0, inv_desc1))

            positive_samples, negative_samples = sample_from_augmented_pair(IMAGE_SHAPE, [batch[sample_index]['augmentor'] for sample_index in range(2)])
            
            if len(positive_samples) == 0 or len(negative_samples) == 0: continue  # TODO: catch in dataloader. 

            # #################### TESTING #####################
            # half_image_dim = tuple([int(round(i/2)) for i in IMAGE_SHAPE])
            # resize = Resize(half_image_dim)
            # resized_descriptors = resize(inv_descriptors)
            # positive_samples = (positive_samples/2).round().long()
            # positive_samples[0] = torch.clamp(positive_samples[0], min=0, max=half_image_dim[0]-1)
            # positive_samples[1] = torch.clamp(positive_samples[1], min=0, max=half_image_dim[1]-1)

            # for i in range(len(negative_samples)):
            #     negative_samples[i] = (negative_samples[i]/2).round().long()
            #     negative_samples[i][0] = torch.clamp(negative_samples[i][0], min=0, max=half_image_dim[0]-1)
            #     negative_samples[i][1] = torch.clamp(negative_samples[i][1], min=0, max=half_image_dim[1]-1)
            ##################################################
            
            
            loss = get_loss(inv_descriptors, positive_samples, negative_samples)
            loss_accumulated.append(torch.tensor(0.0).to(device))
            for k, v in loss.items(): loss_accumulated[-1] += v

            if sample_index % 1 == 0:
                (sum(loss_accumulated) / len(loss_accumulated)).backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_accumulated = []


            logger.update(epoch, model, optimizer, loss, images, descriptors)

            # # if i % 500 == 0:
            # #     #################### Temp VISUALIZE ####################
            # #     ########################################################
            # #     fig = plt.figure()
            # #     gs = fig.add_gridspec(1, 2)
            # #     axs = gs.subplots()
            # #     im = torch.clamp(images[0].clone().detach().cpu().permute(1,2,0).float(), min=0, max=1)
            # #     d = torch.clamp(((descriptors[0]+1)/2).clone().detach().cpu().permute(1,2,0).float(), min=0, max=1)
            # #     axs[0].imshow(im)
            # #     axs[1].imshow(d)
            # #     plt.savefig(f'{checkpoint_dir}/images/{epoch}_{i}')

    logger.save_model(model, optimizer, model_dir_end='_final')
