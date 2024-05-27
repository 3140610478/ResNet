import os
import sys
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

base_folder = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
if base_folder not in sys.path:
    sys.path.append(base_folder)
if True:
    from config import GoogLeNet_training_weights


class ConvBlock(nn.Conv2d):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        torch.nn.init.kaiming_normal_(self.weight)
        self.norm = nn.BatchNorm2d(self.out_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        y = super().forward(input)
        y = self.norm(y)
        return F.relu(y)


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3r, ch3x3, ch5x5r, ch5x5, chpool):
        super().__init__()
        self.branch1x1 = ConvBlock(in_channels, ch1x1, 1, 1, 0)
        self.branch3x3 = nn.Sequential(
            ConvBlock(in_channels, ch3x3r, 1, 1, 0),
            ConvBlock(ch3x3r, ch3x3, 3, 1, 1),
        )
        self.branch5x5 = nn.Sequential(
            ConvBlock(in_channels, ch5x5r, 1, 1, 0),
            ConvBlock(ch5x5r, ch5x5, 5, 1, 2),
        )
        self.branchpool = nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
            ConvBlock(in_channels, chpool, 1, 1, 0)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output1x1 = self.branch1x1(input)
        output3x3 = self.branch3x3(input)
        output5x5 = self.branch5x5(input)
        outputpool = self.branchpool(input)
        output = torch.cat(
            (output1x1, output3x3, output5x5, outputpool), dim=1)
        return output


class GoogLeNet(nn.Module):
    def __init__(self):
        super().__init__()
        return_layers = {
            "8":    "a0",
            "11":   "a1",
            "14":   "c"
        }
        self.backbone = IntermediateLayerGetter(
            nn.Sequential(
                ConvBlock(1, 16, 5, 1, 2),
                nn.MaxPool2d(3, 1, 1),
                ConvBlock(16, 48, 3, 1, 1),
                nn.MaxPool2d(3, 1, 1),

                Inception(48, 16, 24, 32, 4, 8, 8),
                Inception(64, 32, 32, 48, 8, 24, 16),

                nn.MaxPool2d(3, 2, 1),

                Inception(120, 48, 24, 52, 4, 12, 16),
                Inception(128, 40, 28, 56, 6, 16, 16),
                Inception(128, 32, 32, 64, 6, 16, 16),
                Inception(128, 28, 36, 72, 8, 16, 16),
                Inception(132, 64, 40, 80, 8, 32, 32),

                nn.MaxPool2d(3, 2, 1),

                Inception(208, 64, 40, 80,  8, 32, 32),
                Inception(208, 96, 48, 96, 12, 32, 32),
            ),
            return_layers=return_layers,
        )

        # Softmax activation: Removed because Softmax is applied in CrossEntropyLoss
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(1),
            nn.Dropout(0.4),
            nn.Linear(256, 10),
            # nn.Softmax(dim=1),
        )
        self.aux0 = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            ConvBlock(128, 32, 1),
            nn.Flatten(1),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.3),
            nn.Linear(256, 10),
            # nn.Softmax(dim=1),
        )
        self.aux1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            ConvBlock(208, 32, 1),
            nn.Flatten(1),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.3),
            nn.Linear(256, 10),
            # nn.Softmax(dim=1),
        )
        self.training_weights = nn.Parameter(torch.tensor(
            GoogLeNet_training_weights), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)

        c = self.classifier(features["c"])
        if self.training:
            a0 = self.aux0(features["a0"])
            a1 = self.aux1(features["a1"])

            c = torch.stack((c, a0, a1))

        return c


if __name__ == "__main__":
    googlenet = GoogLeNet().to("cuda")
    print(str(googlenet))
    a = torch.zeros((16, 1, 28, 28)).to("cuda")
    print(a.shape, googlenet.forward(a).shape)
    pass
