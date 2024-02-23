import os
import sys
import torch
import plotly.graph_objects as go
from plotly.offline import plot
pass
base_folder = os.path.dirname(os.path.abspath(__file__))
if base_folder not in sys.path:
    sys.path.append(base_folder)
if True:
    from Networks.Networks import ResNet110, NonResNet110, ResNet32, NonResNet32
    from Networks.Training import check
    from Data.CIFAR10 import CIFAR10_test_loader, CIFAR10_classes
    from Log.Logger import getLogger

if __name__ == '__main__':
    resnet110 = torch.load(os.path.abspath(os.path.join(
        base_folder, "./Networks/ResNet110/ResNet110.resnet110")))
    check_logger = getLogger("check_resnet110")
    check_logger.info("Check for ResNet110:\n")
    confusion_matrix = check(resnet110, check_logger)

    fig = go.Figure(
        data=go.Heatmap(
            z=confusion_matrix,
            x=CIFAR10_classes,
            y=CIFAR10_classes,
            text=confusion_matrix,
            texttemplate="%{z:.3f}",
            hoverongaps=False,
        )
    )
    fig.update_layout(
        width=800,
        height=800,
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
    )
    plot(fig, filename=os.path.abspath(os.path.join(base_folder, "./check.html")))
    pass
