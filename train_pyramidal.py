import yaml
import torch
from tqdm import tqdm 
from addict import Dict
from logger import Logger
import pytorch_lightning as pl
from dataset import make_data_loader
from torchvision.transforms import Resize
from models.pyramidal import PyramidalDenseNet
from augmentations.util import image_de_normalize
from pytorch_lightning.loggers import WandbLogger

with open('configuration/train.yaml') as f:
    cfg = Dict(yaml.safe_load(f))
    
dataloader = make_data_loader(split='train', args=cfg.dataloader)

# model
model = PyramidalDenseNet(cfg.model)

# training
trainer = pl.Trainer(accelerator='gpu', gpus=1, precision=16, limit_train_batches=0.5, logger=WandbLogger())

trainer.fit(model, dataloader, dataloader)
	
