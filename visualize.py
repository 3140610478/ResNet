import os
import sys
import plotly.graph_objects as go
from plotly.express.colors import qualitative as colorset
from plotly.offline import plot
from plotly.subplots import make_subplots
pass
base_folder = os.path.dirname(os.path.abspath(__file__))
log_folder = os.path.abspath(os.path.join(base_folder, "./Log/"))
if base_folder not in sys.path:
    sys.path.append(base_folder)
if True:
    from Networks import models
    from Data import datasets


def read_training_log(filename):
    with open(os.path.abspath(os.path.join(log_folder, f"./{filename}"))) as log:
        lines = log.readlines()
    lines = [i.split("\t") for i in lines[4:] if i != "\n"]
    lines = [[j.rstrip().rstrip(",") for j in i] for i in lines if len(i) > 1]
    groups = ("train", "val", "test",)
    contents = {"[loss]": {_: []
                           for _ in groups}, "[acc]": {_: [] for _ in groups}}
    for line in lines:
        content = contents[line[0]]
        for record in line[1:]:
            group, value = record.split(":")
            content[group].append(float(value))
    contents["[len]"] = len(contents["[loss]"]["train"])
    return contents


def plot_contents(fig: go.Figure, name: str, contents: dict[str, dict[str, list]], color: str) -> None:
    length, loss, acc = contents["[len]"], contents["[loss]"], contents["[acc]"]
    x = list(range(1, 1+length))
    fig.add_trace(
        go.Scatter(
            name=name+" train",
            x=x,
            y=loss["train"],
            mode="lines",
            line={"dash": "dot", "color": color, },
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            name=name+" val",
            x=x,
            y=loss["val"],
            mode="lines",
            line={"color": color, },
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            name=name+" test",
            x=[1+length],
            y=loss["test"],
            mode="markers",
            marker={"color": color, }
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            name=name+" train",
            x=x,
            y=acc["train"],
            mode="lines",
            line={"dash": "dot", "color": color, },
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            name=name+" val",
            x=x,
            y=acc["val"],
            mode="lines",
            line={"color": color, },
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            name=name+" test",
            x=[1+length],
            y=acc["test"],
            mode="markers",
            marker={"color": color, }
        ),
        row=1,
        col=2,
    )
    pass


if __name__ == '__main__':
    folder = os.path.abspath(os.path.join(base_folder, f"./result"))
    if not os.path.exists(folder):
        os.mkdir(folder)
    for dataset in datasets:
        fig = make_subplots(rows=1, cols=2, subplot_titles=("[loss]", "[acc]"),
                            horizontal_spacing=0.1)
        colors = colorset.Plotly
        for model, color in zip(models, colors):
            name = name = f"{model.__name__}_on_{dataset.__name__.rsplit('.')[-1]}"
            log = read_training_log(f"./{name}.log")

            plot_contents(fig, f"{model.__name__}", log, color)

        fig.update_xaxes(title_text="epochs", row=1, col=1)
        fig.update_xaxes(title_text="epochs", row=1, col=2)
        fig.update_yaxes(title_text="loss", row=1, col=1)
        fig.update_yaxes(title_text="accuracy", row=1, col=2)
        plot(fig, filename=os.path.abspath(
            os.path.join(base_folder, f"./result/{dataset.__name__.rsplit('.')[-1]}.html")))
    pass
