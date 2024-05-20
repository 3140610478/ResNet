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
    from Data.utils import DatasetTransformer, train_transform, val_transform, test_transform, check_transform, demo_transform
    from Log.Logger import getLogger


data_path = os.path.abspath(os.path.join(
    base_folder, config.data_path, "./MNIST"))
if not os.path.exists(data_path):
    os.mkdir(data_path)


classes = tuple(str(i) for i in range(10))


training_set = torchvision.datasets.MNIST(
    data_path,
    train=True,
    transform=None,
    download=True
)
train_set, val_set = random_split(
    training_set, (54000, 6000), torch.Generator().manual_seed(config.seed)
)

train_set = DatasetTransformer(train_set, train_transform)
val_set = DatasetTransformer(val_set, val_transform)
demo_set = torchvision.datasets.MNIST(
    data_path,
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
    "MNIST Datasets\n32x32 images for 10 classes\nNumber of samples from each class:\n" + \
    "MNIST_len_train, MNIST_len_val, MNIST_len_test = {}, {}, {}\n".format(len_train, len_val, len_test) + \
    "Training Set:\n" + \
    str(one_hot(torch.cat([y for (_, y) in train_loader], dim=0)).sum(dim=0)) + "\n" + \
    "Validating Set:\n" + \
    str(one_hot(torch.cat([y for (_, y) in val_loader], dim=0)).sum(dim=0)) + "\n" + \
    "Testing Set:\n" + \
    str(one_hot(torch.cat([y for (_, y) in test_loader], dim=0)).sum(dim=0)) + "\n" + \
    "\n"


logger = getLogger("MNIST")
logger.info(INFO)


if __name__ == "__main__":
    for x, y in check_loader:
        print(x.shape, y.shape)
