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


class TinyVGG(nn.Module):
    """
    Tiny pet classification model for testing purposes..
    """

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_channels * 56 * 56, out_features=out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)

        return x
