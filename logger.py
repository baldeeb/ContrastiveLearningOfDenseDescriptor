import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


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
