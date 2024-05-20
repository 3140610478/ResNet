import os
import sys
import torch
from torchvision import transforms

base_folder = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
if base_folder not in sys.path:
    sys.path.append(base_folder)
if True:
    import config

class ToDevice(torch.nn.Module):
    def __init__(self, device=config.device):
        super().__init__()
        self.device = device

    def forward(self, input: torch.Tensor):
        return input.to(self.device)


class DatasetTransformer(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset, transform) -> None:
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.dataset[index]
        x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.dataset)


train_transform = transforms.Compose([
    transforms.ToTensor(),
    ToDevice(),
    transforms.RandomCrop(28, padding=4),
    transforms.RandomHorizontalFlip(),
])
val_transform = transforms.Compose([
    transforms.ToTensor(),
    ToDevice(),
])
test_transform = val_transform
check_transform = test_transform
demo_transform = check_transform
