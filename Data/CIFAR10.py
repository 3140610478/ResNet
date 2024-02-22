import os
import sys
import torch
import torchvision
from torchvision import transforms
from torch.nn.functional import one_hot
from torch.utils.data import Dataset, DataLoader, random_split
base_folder = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
if base_folder not in sys.path:
    sys.path.append(base_folder)

data_path = os.path.abspath(os.path.join(base_folder, "./Data/CIFAR10"))
batch_size = 128


class DatasetTransformer(Dataset):
    def __init__(self, dataset: Dataset, transform) -> None:
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index)->tuple[torch.Tensor, torch.Tensor]:
        x, y = self.dataset[index]
        return self.transform(x), y

    def __len__(self):
        return len(self.dataset)


train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
])
val_transform = transforms.Compose([
    transforms.ToTensor(),
])
test_transform = val_transform


training_set = torchvision.datasets.CIFAR10(
    data_path,
    train=True,
    transform=None,
    download=True
)
CIFAR10_train, CIFAR10_val = random_split(training_set, (45000, 5000))

CIFAR10_train = DatasetTransformer(CIFAR10_train, train_transform)
CIFAR10_val = DatasetTransformer(CIFAR10_val, val_transform)
CIFAR10_test = torchvision.datasets.CIFAR10(
    data_path,
    train=False,
    transform=test_transform,
    download=True
)

CIFAR10_len_train, CIFAR10_len_val, CIFAR10_len_test = \
    len(CIFAR10_train), len(CIFAR10_val), len(CIFAR10_test)

CIFAR10_train_loader = DataLoader(
    CIFAR10_train, batch_size=batch_size, shuffle=True)
CIFAR10_val_loader = DataLoader(
    CIFAR10_val, batch_size=batch_size, shuffle=True)
CIFAR10_test_loader = DataLoader(
    CIFAR10_test, batch_size=batch_size, shuffle=True)

CIFAR10_info = "CIFAR10 Datasets\n32x32 images for 10 classes\nNumber of samples from each class:\n" + \
    "CIFAR10_len_train, CIFAR10_len_val, CIFAR10_len_test = {}, {}, {}\n".format(CIFAR10_len_train, CIFAR10_len_val, CIFAR10_len_test) + \
    "Training Set:\n" + \
    str(one_hot(torch.cat([y for (_, y) in CIFAR10_train_loader], dim=0)).sum(dim=0)) + "\n" + \
    "Validating Set:\n" + \
    str(one_hot(torch.cat([y for (_, y) in CIFAR10_val_loader], dim=0)).sum(dim=0)) + "\n" + \
    "Testing Set:\n" + \
    str(one_hot(torch.cat([y for (_, y) in CIFAR10_test_loader], dim=0)).sum(dim=0)) + "\n" + \
    "\n"
