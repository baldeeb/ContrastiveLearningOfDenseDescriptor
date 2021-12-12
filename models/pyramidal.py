
import torch
from torch import nn
from torchvision import transforms
import pytorch_lightning as pl
from loss.pyramidal_loss import pyramidal_contrastive_augmentation_loss as ploss
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

# TODO: consolidate into a single logger
from logger import Logger  
import wandb
# TODO: remove
import matplotlib.pyplot as plt  

class PyramidalDenseNet(pl.LightningModule):

	def __init__(self, cfg):
		super().__init__()

		if not self.__config_valid(cfg):
			exit(1)

		deconv_kernel = 4
		backbone = resnet_fpn_backbone(cfg['backbone_name'], pretrained=True)

		for param in backbone.parameters():
			param.requires_grad = True

		self.fpn = torch.nn.Sequential(*(list(backbone.children()))).to(cfg['device'])
		transpose_conv = nn.ConvTranspose2d(cfg['feature_dim'], 
										cfg['feature_dim']//4, 
										deconv_kernel, 
										stride=2, 
										padding=deconv_kernel // 2 - 1, 
										bias=False)
		transpose_batchnorm = nn.BatchNorm2d(cfg['feature_dim']//4)
		final_conv = nn.Conv2d(cfg['feature_dim'] // 4, cfg['output_dim'], 3, 1, 1)

		self.transpose_head = nn.Sequential(transpose_conv, 
										transpose_batchnorm, 
										final_conv).to(cfg['device'])

		for param in self.transpose_head.parameters():
			param.requires_grad = True
		
		# # TODO REMOVE
		# self.logger__ = Logger(model_save_rate=1000000000) # TODO: specify data mean and std
		# print(f"storing this run in: {self.logger__.checkpoint_dir}")


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
		return self.__full_forward(images)[0]

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
		return optimizer

	def training_step(self, train_batch, batch_idx):
		images, metas = train_batch
		z = self.__full_forward(images)

		# self.save_visuals(images, z[0], metas[0]['augmentor'].de_normalize)

		loss_dict = ploss(z, metas[0])
		for k, v in loss_dict.items():
			avg_v = sum(v)/len(v)
			loss_dict[k] = avg_v
			self.log(k, avg_v)

		# TODO: REMOVE
		# for k, v in loss_dict.items():
		# 	self.log(k, v)
		# self.logger__.update(0, None, None, loss_dict, images, z[0])
		# self.logger__.save_model(model, optimizer, model_dir_end='_final')
		
		if batch_idx % 20 == 0:	
			d = z[0][0].clone().detach()
			d = torch.clamp(d.sigmoid(), min=0, max=1)
			plt.imshow(d.permute(1,2,0).cpu().float().numpy())
			plt.savefig(f"temp/images/{self.current_epoch}_{batch_idx}.png")
			# im = plt.imshow(d.permute(1,2,0).cpu().float().numpy())
			# wandb_im = wandb.Image(im, caption="Descriptor")
			# self.log('descriptor', wandb_im)
			# self.log({'descriptor': [wandb.Image(d, caption="Descriptor")]})

		loss_l = list(loss_dict.values())
		loss = sum(loss_l)/len(loss_l)
		return loss

	# def validation_step(self, val_batch, batch_idx):
	# 	pass
	
	# def training_step_end(self, outs):
	# 	pass




# class WandbImagePredCallback(pl.Callback):
#     """Logs the input images and output predictions of a module.
    
#     Predictions and labels are logged as class indices."""
    
#     def __init__(self, val_samples, num_samples=32):
#         super().__init__()
#         self.val_imgs, self.val_labels = val_samples
#         self.val_imgs = self.val_imgs[:num_samples]
#         self.val_labels = self.val_labels[:num_samples]
          
#     def training_step_end(self, trainer, pl_module):
#         val_imgs = self.val_imgs.to(device=pl_module.device)

#         logits = pl_module(val_imgs)
#         preds = torch.argmax(logits, 1)

#         trainer.logger.experiment.log({
#             "val/examples": [
#                 wandb.Image(x, caption=f"Pred:{pred}, Label:{y}") 
#                     for x, pred, y in zip(val_imgs, preds, self.val_labels)
#                 ],
#             "global_step": trainer.global_step
#             })

# 	# def save_visuals(self, images, descriptors, de_normalizer=None):
# 	# 	im = images.clone().detach()
# 	# 	if de_normalizer is None:
# 	# 		im = de_normalizer(im)
# 	# 	im = torch.clamp(im[0], min=0, max=1)
# 	# 	self.log('image', im)
# 	# 	d = descriptors[0].clone().detach()
# 	# 	d = torch.clamp(d.sigmoid(), min=0, max=1)
# 	# 	self.log('descriptor', d)


