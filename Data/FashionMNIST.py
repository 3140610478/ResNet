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
if True:
    import config

if not os.path.exists(config.data_path):
    os.mkdir(config.data_path)


class ToDevice(torch.nn.Module):
    def __init__(self, device=config.device):
        super().__init__()
        self.device = device
    
    def forward(self, input: torch.Tensor):
        return input.to(self.device)


class DatasetTransformer(Dataset):
    def __init__(self, dataset: Dataset, transform) -> None:
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.dataset[index]
        x = self.transform(x)
        x = torch.cat((x, x, x), dim=0)
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


classes = ("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")


training_set = torchvision.datasets.FashionMNIST(
    config.data_path,
    train=True,
    transform=None,
    download=True
)
train_set, val_set = random_split(
    training_set, (54000, 6000))

train_set = DatasetTransformer(train_set, train_transform)
val_set = DatasetTransformer(val_set, val_transform)
demo_set = torchvision.datasets.FashionMNIST(
    config.data_path,
    train=False,
    transform=None,
    download=True
)
test_set = DatasetTransformer(demo_set, test_transform)

len_train, len_val, len_test = \
    len(train_set), len(val_set), len(test_set)

train_loader = DataLoader(
    train_set, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(
    val_set, batch_size=config.batch_size, shuffle=True)
test_loader = DataLoader(
    test_set, batch_size=config.batch_size, shuffle=True)
check_loader = test_loader

INFO = \
    "FashionMNIST Datasets\n32x32 images for 10 classes\nNumber of samples from each class:\n" + \
    "FashionMNIST_len_train, FashionMNIST_len_val, FashionMNIST_len_test = {}, {}, {}\n".format(len_train, len_val, len_test) + \
    "Training Set:\n" + \
    str(one_hot(torch.cat([y for (_, y) in train_loader], dim=0)).sum(dim=0)) + "\n" + \
    "Validating Set:\n" + \
    str(one_hot(torch.cat([y for (_, y) in val_loader], dim=0)).sum(dim=0)) + "\n" + \
    "Testing Set:\n" + \
    str(one_hot(torch.cat([y for (_, y) in test_loader], dim=0)).sum(dim=0)) + "\n" + \
    "\n"

if __name__ == "__main__":
    for x, y in check_loader:
        print(x.shape, y.shape)
