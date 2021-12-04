
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl


from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class PyramidalDenseNet(pl.LightningModule):

	def __init__(self):
		super().__init__()
		cfg = { 'backbone_name': 'wide_resnet50_2', 
				'feature_dim': 256, 
				'input_dim': 3, 
				'output_dim': 4,
				'device': 'cpu' }
		if not self.__config_valid(cfg): exit(1)
		
		deconv_kernel = 4
		backbone = resnet_fpn_backbone(cfg['backbone_name'], pretrained=True)

		for param in backbone.parameters():
			param.requires_grad = True

		self.fpn = torch.nn.Sequential(*(list(backbone.children()))).to(cfg['device'])
		transpose_conv = nn.ConvTranspose2d(cfg['feature_dim'], cfg['feature_dim']//4, deconv_kernel, stride=2, padding=deconv_kernel // 2 - 1, bias=False)
		transpose_batchnorm = nn.BatchNorm2d(cfg['feature_dim']//4)
		final_conv = nn.Conv2d(cfg['feature_dim'] // 4, cfg['output_dim']+1, 3, 1, 1)
		self.transpose_head = nn.Sequential(transpose_conv, transpose_batchnorm, final_conv).to(cfg['device'])

		for param in self.transpose_head.parameters():
			param.requires_grad = True

	def __config_valid(self, cfg):
		if cfg['input_dim'] != 3:
			print('WARNING: the feature pyramid net backbone is only set up for RGB input.')
			raise IOError
		return True

	def __full_forward(self, images):
		z = self.fpn(images)
		z_hat = self.transpose_head(z[0])
		return torch.cat([z, z_hat], dim=0)

	def forward(self, x):
		return self.__full_forward(images)[0]

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
		return optimizer

	def training_step(self, train_batch, batch_idx):
		# x, y = train_batch  # This is temporary
		# images = torch.rand(2, 3, 49, 49).double()
		
		images, metas = train_batch
		z = self.__full_forward(images)
		
		
		loss_dict = self.loss(z)

		for k,v in loss_dict.items(): self.log(k, v)
		loss = sum(list(loss_dict.values()))
		return loss

	def validation_step(self, val_batch, batch_idx):
		x, y = val_batch
		x = x.view(x.size(0), -1)
		# z = self.encoder(x)
		# x_hat = self.decoder(z)
		# loss = F.mse_loss(x_hat, x)
		# self.log('val_loss', loss)

		

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
	
