import os
import sys
import torch

import Networks.Training
pass
base_folder = os.path.dirname(os.path.abspath(__file__))
if base_folder not in sys.path:
    sys.path.append(base_folder)
if True:
    import Networks
    import Data
    from Log.Logger import getLogger
    from config import device

criterion = torch.nn.CrossEntropyLoss()


def run(model, data, name=None):
    if not isinstance(name, str):
        name = f"{model.__class__.__name__}_on_{data.__name__.rsplit('.')[-1]}"

    path = os.path.abspath(os.path.join(
        base_folder, "./Networks/save"
    ))
    if not os.path.exists(path):
        os.mkdir(path)
    logger = getLogger(name)

    optimizers = Networks.Training.get_optimizers(model)

    # ResNet20
    logger.info(f"{name}\n")
    # warm-up epoch
    logger.info("Warm-up")
    start_epoch = 0
    start_epoch = Networks.Training.train_until(
        model, data, logger, 0.2, optimizers[0.01]
    )
    logger.info("learning_rate = 0.1")
    Networks.Training.train_epoch_range(
        model, data, logger, start_epoch, 10, optimizers[0.1]
    )
    logger.info("learning_rate = 0.01")
    Networks.Training.train_epoch_range(
        model, data, logger, 10, 15, optimizers[0.01]
    )
    logger.info("learning_rate = 0.001")
    Networks.Training.train_epoch_range(
        model, data, logger, 15, 20, optimizers[0.001]
    )
    Networks.Training.test(model, data, logger)
    torch.save(
        model,
        os.path.abspath(os.path.join(
            path, f"./{name}.model"
        )),
    )


if __name__ == "__main__":
    for model in Networks.models:
        for dataset in Data.datasets:
            run(model().to(device), dataset)
    pass
