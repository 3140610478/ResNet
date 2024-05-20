import os
import sys
import torch
import plotly.graph_objects as go
from plotly.offline import plot

import Networks.Training
pass
base_folder = os.path.dirname(os.path.abspath(__file__))
if base_folder not in sys.path:
    sys.path.append(base_folder)
if True:
    import Networks
    import Data
    from Log.Logger import getLogger

if __name__ == '__main__':
    folder = os.path.abspath(os.path.join(base_folder, f"./result"))
    if not os.path.exists(folder):
        os.mkdir(folder)
    for dataset in Data.datasets:
        folder = os.path.abspath(os.path.join(base_folder, f"./result/{dataset.__name__.rsplit('.')[-1]}"))
        if not os.path.exists(folder):
            os.mkdir(folder)
        for model in Networks.models:
            name = f"{model.__name__}_on_{dataset.__name__.rsplit('.')[-1]}"
            m = torch.load(os.path.abspath(os.path.join(
                base_folder, f"./Networks/save/{name}.model")))
            
            check_logger = getLogger(f"Check_{name}")
            check_logger.info(f"Check for {name}:\n")
            confusion_matrix = Networks.Training.check(m, dataset, check_logger)
            
            acc = (torch.eye(10) * confusion_matrix).sum() / 10

            fig = go.Figure(
                data=go.Heatmap(
                    z=confusion_matrix,
                    x=dataset.classes,
                    y=dataset.classes,
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
            plot(fig, filename=os.path.abspath(os.path.join(folder, f"./Check_{name}.html")))
            
    pass
