import os
import sys

base_folder = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
if base_folder not in sys.path:
    sys.path.append(base_folder)
if True:
    from Networks.MLP import MLP
    from Networks.LeNet import LeNet
    from Networks.AlexNet import AlexNet
    from Networks.GoogLeNet import GoogLeNet
    from Networks.ResNet20 import ResNet20

models = (MLP, LeNet, AlexNet, GoogLeNet, ResNet20)
