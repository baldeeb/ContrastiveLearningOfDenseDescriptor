
import torch
from torch import nn
from torchvision import transforms
import pytorch_lightning as pl
from loss.pyramidal_loss import pyramidal_contrastive_augmentation_loss as ploss
from loss.pyramidal_loss import non_contrastive_loss as nloss
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

import wandb


class PyramidalDenseNet(pl.LightningModule):

	def __init__(self, cfg):
		super().__init__()
		if not self.__config_valid(cfg):
			exit(1)
		deconv_kernel = 4

		# Setup Feature Pyramid Net
		backbone = resnet_fpn_backbone(cfg['backbone_name'], pretrained=True)
		for param in backbone.parameters():
			param.requires_grad = True
		self.fpn = torch.nn.Sequential(
			*(list(backbone.children()))).to(cfg['device'])

		# Setup Trans Conv Head
		transpose_conv = nn.ConvTranspose2d(cfg['feature_dim'],
										cfg['feature_dim']//4,
										deconv_kernel,
										stride=2,
										padding=deconv_kernel // 2 - 1,
										bias=False)
		transpose_batchnorm = nn.BatchNorm2d(cfg['feature_dim']//4)
		final_conv = nn.ConvTranspose2d(cfg['feature_dim'] // 4,
										cfg['output_dim'],
										deconv_kernel,
										stride=2,
										padding=deconv_kernel // 2 - 1,
										bias=False)
		self.transpose_head = nn.Sequential(transpose_conv,
										transpose_batchnorm,
										final_conv).to(cfg['device'])
		for param in self.transpose_head.parameters():
			param.requires_grad = True

		# Projection Head
		# This is intended for the lowest dimension desriptor in the pyramid 
		low_feat_dim = 80
		self.prediction_head = nn.Sequential(
			nn.Flatten(),
			nn.Linear(low_feat_dim, low_feat_dim),
			nn.LayerNorm(low_feat_dim),
			nn.Linear(low_feat_dim, low_feat_dim)
		)
		for param in self.prediction_head.parameters():
			param.requires_grad = True

	def __config_valid(self, cfg):
		if cfg['input_dim'] != 3:
			print('WARNING: the feature pyramid net backbone is only set up for RGB input.')
			raise IOError
		return True

	def __full_forward(self, images):
		z = self.fpn(images)
		z_hat = self.transpose_head(z['0'])
		return [z_hat, *list(z.values())]

	def forward(self, x):
		return self.__full_forward(x)[0]

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
		return optimizer

	def training_step(self, train_batch, batch_idx):
		self.batch_idx = batch_idx
		self.ts_images, self.ts_meta = train_batch
		self.ts_results = self.__full_forward(self.ts_images)

		# Pixel Contrastive Loss
		if False: # NOTE: This is temporary
			loss_dict = ploss(self.ts_results, self.ts_meta[0])
			for k, v in loss_dict.items():
				avg_v = sum(v)/len(v)
				loss_dict[k] = avg_v
				self.log(k, avg_v)
			loss_l = list(loss_dict.values())
			loss = sum(loss_l)/len(loss_l)

		# Non-contrastive loss
		if False: # NOTE: This is temporary
			# TODO: before this is done, one needs to de-augment since this is non-iconic data.
			z1, z2 = self.ts_results[-1][0], self.ts_results[-1][1]
			z1, z2 = nn.Flatten()(z1), nn.Flatten()(z2)
			p1, p2 = self.prediction_head(z1), self.prediction_head(z2)
			loss = (nloss(p1, z1) + nloss(p2, z2)) / 2
			self.log('non-contrastive', loss)

		return loss

	def configure_optimizers(self):
		'''
		TODO: try reduced learning rate and use faster learning rates for the prediction head
		'''
		return torch.optim.Adam(self.parameters(), lr=0.001)



class WandbImageCallback(pl.Callback):
	"""Logs the input images and output predictions of a module.

	Predictions and labels are logged as class indices."""

	def __init__(self, rate=20):
		super().__init__()
		self.rate = rate 

	def on_batch_end(self, trainer, pl_module):
		if trainer.global_step % self.rate != 0: return
		log_dict = {"global_step": trainer.global_step}
		
		# log image
		im = pl_module.ts_meta[0]['image'].clone().detach()
		log_dict['visuals/image'] =  wandb.Image(im)
				
		# log descriptor
		d = pl_module.ts_results[0].clone().detach()
		d = torch.clamp(d.sigmoid(), min=0, max=1)
		log_dict['visuals/descriptor'] = wandb.Image(d)

		trainer.logger.experiment.log(log_dict)


