import yaml
import torch
from tqdm import tqdm 
from addict import Dict
from logger import Logger
from dataset import make_data_loader
from models.dense_model import DenseModel
from torchvision.transforms import Resize
from util.model_storage import load_dense_model
from augmentations.util import image_de_normalize
from loss.augmentation_loss import contrastive_augmentation_loss, overlapping_region_positive_sample_loss


# Read Config
with open('configuration/train.yaml') as f: cfg = Dict(yaml.safe_load(f))

# Dataloader
dataloader = make_data_loader(split='train', args=cfg.dataloader)

# Model Setup
if cfg.load_model_path:
    model, optimizer = load_dense_model(cfg.load_model_path)
else:
    model = DenseModel(cfg.model.output_dim, False, device=cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Setup Logger
# TODO: switch to wandb
logger = Logger(image_de_normalizer=image_de_normalize) # TODO: specify data mean and std
print(f"Checkpoints stored in: {logger.checkpoint_dir}")

loss_accumulated = []
IMAGE_SHAPE = dataloader.dataset[0]['image'].shape[-2:]

for epoch in range(cfg.dataloader.num_epochs):
    for sample_index, (images, meta) in enumerate(tqdm(dataloader)):

        images = images.to(cfg.device)
        descriptors = model(images)
        
        # ##############################
        # # For testing. 
        # # Answering the question: How much do masks (informed sampling) help?
        # mask = dataloader.dataset[meta[0]['index']]['classmask']
        # mask[mask != 0] = 1
        # loss = contrastive_augmentation_loss(descriptors, meta, mask)
        # ################################

        loss = contrastive_augmentation_loss(descriptors, meta)

        # ###############################
        # # For Testing
        # # using all positive samples in loss
        loss['match'] = overlapping_region_positive_sample_loss(descriptors, meta)
        # ###############################

        loss_accumulated.append(torch.tensor(0.0).to(cfg.device))
        for k, v in loss.items(): loss_accumulated[-1] += v

        mean_loss = sum(loss_accumulated) / len(loss_accumulated)
        if mean_loss != 0:
            mean_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_accumulated = []
            
            logger.update(epoch, model, optimizer, loss, images, descriptors)

logger.save_model(model, optimizer, model_dir_end='_final')
