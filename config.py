import os
import torch

base_folder = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.abspath(os.path.join(base_folder, "./Data/FashionMNIST"))
batch_size = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
