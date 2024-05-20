import os
import sys

base_folder = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
if base_folder not in sys.path:
    sys.path.append(base_folder)
if True:
    from Data import MNIST, FashionMNIST
    
datasets = (MNIST, FashionMNIST)