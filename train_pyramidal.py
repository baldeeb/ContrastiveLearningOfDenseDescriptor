from tqdm import tqdm 
from logger import Logger

import torch
from torchvision.transforms import Resize

from loss import contrastive_augmentation_loss
from models.dense_model import DenseModel
from augmentations.util import image_de_normalize
from dataset import make_data_loader
	

# data
dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
mnist_train, mnist_val = random_split(dataset, [55000, 5000])
train_loader = DataLoader(mnist_train, batch_size=32)
val_loader = DataLoader(mnist_val, batch_size=32)


# model
model = PyramidalDenseNet()

# training
trainer = pl.Trainer(precision=16, limit_train_batches=0.5)
# trainer = pl.Trainer(gpus=4, num_nodes=8, precision=16, limit_train_batches=0.5)
trainer.fit(model, train_loader, val_loader)
	
