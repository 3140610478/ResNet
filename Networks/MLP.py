import torch
from torch import nn


class MLP(nn.Sequential):
    def __init__(self):
        layers = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10),
            # # Softmax activation: Removed because Softmax is applied in CrossEntropyLoss
            # nn.Softmax(dim=1),
        )
        super().__init__(*layers)

if __name__ == "__main__":
    mlp = MLP().to("cuda")
    print(str(mlp))
    while True:
        a = torch.zeros((16, 1, 28, 28)).to("cuda")
        print(a.shape, mlp.forward(a).shape)
    pass
