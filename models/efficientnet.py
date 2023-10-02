# Created by Baole Fang at 8/30/23

from models import register
import torch
import torch.nn as nn
import torchvision


@register('efficientnet')
class EfficientNet(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.backbone=torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1).features
        self.pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier=nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=512, bias=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=256, out_features=n, bias=True),
        )
        self.activation=nn.Sigmoid()

    def forward(self,x):
        x=self.backbone(x)
        x=self.pool(x)
        x=torch.flatten(x, 1)
        x=self.classifier(x)
        return x

