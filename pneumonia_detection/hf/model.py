"""
Builds Pytorch model
"""
import torch
import torchvision.models
from torch import nn


class ResNet101(nn.Module):
    """
    ResNet101 model specified for the binary problem. The according transforms were taken from pytorch.org.
    """

    def __init__(self):
        super().__init__()
        self.weights = torchvision.models.ResNet101_Weights.DEFAULT
        self.transforms = self.weights.transforms
        self.resnet = torchvision.models.resnet101(weights=self.weights)

        for param in self.resnet.parameters():
            param.requires_grad = False

        self.resnet.fc = nn.Linear(in_features=2048, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet(x)
        return x
