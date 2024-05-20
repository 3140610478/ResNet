from typing import Iterable
import copy
import torch
from torch import nn


class NormedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        torch.nn.init.kaiming_normal_(self.weight)
        self.norm = nn.BatchNorm2d(self.out_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        y = super().forward(input)
        y = self.norm(y)
        return y


class ResBlock(nn.Module):
    def __init__(self, layers: Iterable[NormedConv2d], reshape: nn.Module | None = None) -> None:
        super().__init__()
        relu = nn.ReLU()
        self.network = []
        for layer in layers:
            self.network.extend((layer, relu))
        self.network = nn.Sequential(*self.network[:-1])
        self.reshape = reshape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.network(x)
        res = x.clone() if self.reshape is None else self.reshape(x)
        return nn.functional.relu(y + res, )


class ConvSequence(nn.Module):
    def __init__(self, layers: int, block: nn.Module, first_block: nn.Module | None = None):
        super().__init__()
        self.network = []
        if first_block is not None:
            self.network.append(first_block)
            layers -= 1
        self.network.extend((copy.deepcopy(block) for _ in range(layers)))
        self.network = nn.Sequential(*self.network)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ResNet20(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # conv (input)
            NormedConv2d(1, 16, 3, padding=1),
            # conv (16 channels)
            ConvSequence(
                3,
                ResBlock(
                    (
                        NormedConv2d(16, 16, 3, padding=1),
                        NormedConv2d(16, 16, 3, padding=1),
                    ),
                ),
            ),
            # conv (32 channels)
            ConvSequence(
                3,
                ResBlock(
                    (
                        NormedConv2d(32, 32, 3, padding=1),
                        NormedConv2d(32, 32, 3, padding=1),
                    ),
                ),
                first_block=ResBlock(
                    (
                        NormedConv2d(16, 32, 3, stride=2, padding=1),
                        NormedConv2d(32, 32, 3, padding=1),
                    ),
                    reshape=nn.Conv2d(16, 32, 1, stride=2, padding=0),
                ),
            ),            
            # conv (64 channels)
            ConvSequence(
                3,
                ResBlock(
                    (
                        NormedConv2d(64, 64, 3, padding=1),
                        NormedConv2d(64, 64, 3, padding=1),
                    ),
                ),
                first_block=ResBlock(
                    (
                        NormedConv2d(32, 64, 3, stride=2, padding=1),
                        NormedConv2d(64, 64, 3, padding=1),
                    ),
                    reshape=nn.Conv2d(32, 64, 1, stride=2, padding=0),
                ),
            ),
            # Average pooling
            nn.AdaptiveAvgPool2d(output_size=1),
            # Flattening
            nn.Flatten(),
            # Fully connected layer
            nn.Linear(64, 10),
            # # Softmax activation: Removed because Softmax is applied in CrossEntropyLoss
            # nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    

if __name__ == "__main__":
    resnet = ResNet20().to("cuda")
    print(str(resnet))
    a = torch.zeros((16, 1, 28, 28)).to("cuda")
    print(a.shape, resnet.forward(a).shape)
    pass