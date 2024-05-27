import torch
from torch import nn


class AlexNet(nn.Sequential):
    def __init__(self, dropout: float = 0.5):
        layers = (
            nn.Conv2d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(16, 48, kernel_size=5, padding=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),

            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(start_dim=1),
            nn.Dropout(p=dropout),
            nn.Linear(64 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 10),
            # # Softmax activation: Removed because Softmax is applied in CrossEntropyLoss
            # nn.Softmax(dim=1)
        )
        super().__init__(*layers)


if __name__ == "__main__":
    alexnet = AlexNet().to("cuda")
    print(str(alexnet))
    while True:
        a = torch.zeros((16, 1, 28, 28)).to("cuda")
        print(a.shape, alexnet.forward(a).shape)
    pass
