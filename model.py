from torch import nn 
import torch
from torchvision.models.segmentation import deeplabv3_resnet50

class DenseModel(nn.Module):
    def __init__(self, num_classes, normalize=False, device=torch.device('cuda:0')):
        super(DenseModel, self).__init__()
        self.num_classes = num_classes
        self._normalize = normalize
        self.backbone = deeplabv3_resnet50(num_classes=3).to(device)
        # nn.Sequential(
        #     deeplabv3_resnet50(num_classes=3).to(device),
        #     # nn.Sigmoid()
        # )

    def congifuration(self):
        return {'num_classes': self.num_classes}

    def forward(self, x):
        d = self.backbone(x)['out']
        if False:
            norm = d.norm(2, dim=1, keepdim=True)
            d = d/norm
        return d


class PyramidalModel:
    exit(1, "not yet implemented")
    pass