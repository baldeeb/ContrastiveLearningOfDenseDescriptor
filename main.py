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
from logger import Logger

import torch
from torchvision.transforms import Resize

from loss import contrastive_augmentation_loss
from model import DenseModel
from util.model_storage import save_model
from augmentations.util import image_de_normalize
from dataset import make_data_loader, sample_from_augmented_pair



if __name__ == '__main__':
    device = torch.device(cfg.device)

    dataloader = make_data_loader(split='train', args=cfg())
    model = DenseModel(3, False, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    logger = Logger(image_de_normalizer=image_de_normalize) # TODO: specify data mean and std
    print(f"storing this run in: {logger.checkpoint_dir}")


    loss_accumulated = []
    IMAGE_SHAPE = (480, 640)

    for epoch in range(50):
        for sample_index, (images, metas) in enumerate(tqdm(dataloader)):

            images = images.to(device)
            descriptors = model(images)
            
            ##############################
            # For testing. 
            # Answering the question: Would masks (informed sampling) help?
            mask = dataloader.dataset[metas[0]['index']]['classmask']
            mask[mask != 0] = 1
            ################################

            loss = contrastive_augmentation_loss(descriptors, metas, mask)
            
            loss_accumulated.append(torch.tensor(0.0).to(device))
            for k, v in loss.items(): loss_accumulated[-1] += v

            if sample_index % 1 == 0:
                (sum(loss_accumulated) / len(loss_accumulated)).backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_accumulated = []
            logger.update(epoch, model, optimizer, loss, images, descriptors)

    logger.save_model(model, optimizer, model_dir_end='_final')
