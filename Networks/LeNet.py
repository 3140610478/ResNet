import torch
from torch import nn

class LeNet(nn.Sequential):
    def __init__(self):
        # for input with one channel and shape of (28, 28) 
        layers = (
            nn.Conv2d(1, 6, 5),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # (_, 6, 12, 12)
            
            nn.Conv2d(6, 16, 5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # (_, 16, 4, 4)
            
            nn.Flatten(start_dim=1),
            nn.Linear(256, 80),
            nn.ReLU(),
            nn.Linear(80, 56),
            nn.ReLU(),
            nn.Linear(56, 10),
            # # Softmax activation: Removed because Softmax is applied in CrossEntropyLoss
            # nn.Softmax(dim=1)
        )
        super().__init__(*layers)
        

if __name__ == "__main__":
    lenet = LeNet().to("cuda")
    print(str(lenet))
    while True:
        a = torch.zeros((16, 1, 28, 28)).to("cuda")
        print(a.shape, lenet.forward(a).shape)
    pass