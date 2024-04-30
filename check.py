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
    from Networks.Networks import ResNet32
    from Networks.Training import check
    from Data.FashionMNIST import test_loader, classes
    from Log.Logger import getLogger

if __name__ == '__main__':
    resnet32 = torch.load(os.path.abspath(os.path.join(
        base_folder, "./Networks/ResNet32/ResNet32.resnet32")))
    check_logger = getLogger("check_resnet32")
    check_logger.info("Check for ResNet32:\n")
    confusion_matrix = check(resnet32, check_logger)
    acc = (torch.eye(10) * confusion_matrix).sum() / 10

    fig = go.Figure(
        data=go.Heatmap(
            z=confusion_matrix,
            x=classes,
            y=classes,
            text=confusion_matrix,
            texttemplate="%{z:.3f}",
            hoverongaps=False,
        )
    )
    fig.update_layout(
        width=800,
        height=800,
        title=f"Confusion Matrix (Overall ACC = {acc:.3f})",
        xaxis_title="Predicted",
        yaxis_title="Actual",
    )
    plot(fig, filename=os.path.abspath(os.path.join(base_folder, "./check.html")))
    pass
