import os
import sys
from PIL import Image
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
    base_folder, config.data_path, "./HWDB"))
if not os.path.exists(data_path):
    os.mkdir(data_path)


class _HWDB_Dataset(Dataset):
    # The file structure was reorganized as follows:
    #     ResNet/Data/HWDB:
    #         train:
    #             label.txt
    #             0000.png
    #             0001.png
    #             .
    #             .
    #             .
    #             xxxx.png
    #         test:
    #             label.txt
    #             0000.png
    #             0001.png
    #             .
    #             .
    #             .
    #             xxxx.png
    def __init__(self, path, fold=1):
        self.path = path
        self.fold = fold
        self.num_samples = len([file for file in os.listdir(
            path) if file.endswith('.png')])

        label = os.path.abspath(os.path.join(path, "./label.txt"))
        with open(label, 'r') as f:
            label = f.read()
            self.LABEL = tuple(int(i) for i in label)

    def __len__(self):
        return self.fold * self.num_samples

    def __getitem__(self, index):
        index = index % self.num_samples
        file = os.path.abspath(os.path.join(self.path, f"000{index}.png"[-8:]))
        return Image.open(file).convert("L"), self.LABEL[index]


classes = tuple("一丁七万丈三上下不与")


training_set = _HWDB_Dataset(os.path.abspath(
    os.path.join(data_path, "./train")), 22)
train_set, val_set = random_split(
    training_set,
    (int(0.9*len(training_set)), int(0.1*len(training_set))),
    torch.Generator().manual_seed(config.seed),
)

train_set = DatasetTransformer(train_set, train_transform)
val_set = DatasetTransformer(val_set, val_transform)

demo_set = _HWDB_Dataset(os.path.abspath(os.path.join(data_path, "./test")))
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
    "HWDB Datasets\nimages with different shapes for 10 classes\nNumber of samples from each class:\n" + \
    "HWDB_len_train, HWDB_len_val, HWDB_len_test = {}, {}, {}\n".format(len_train, len_val, len_test) + \
    "Training Set:\n" + \
    str(one_hot(torch.cat([y for (_, y) in train_loader], dim=0)).sum(dim=0)) + "\n" + \
    "Validating Set:\n" + \
    str(one_hot(torch.cat([y for (_, y) in val_loader], dim=0)).sum(dim=0)) + "\n" + \
    "Testing Set:\n" + \
    str(one_hot(torch.cat([y for (_, y) in test_loader], dim=0)).sum(dim=0)) + "\n" + \
    "\n"


logger = getLogger("HWDB")
logger.info(INFO)


if __name__ == "__main__":
    for x, y in check_loader:
        print(x.shape, y.shape)

