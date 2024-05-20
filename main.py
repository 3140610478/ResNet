import os
import sys
import torch
pass
base_folder = os.path.dirname(os.path.abspath(__file__))
if base_folder not in sys.path:
    sys.path.append(base_folder)
if True:
    from Networks.Networks import ResNet20
    from Networks.Training import get_optimizers, train_epoch_range, train_until, test
    from Data.FashionMNIST import INFO
    from Log.Logger import getLogger
    from config import device
    
criterion = torch.nn.CrossEntropyLoss()


resnet20_path = os.path.abspath(os.path.join(
    base_folder, "./Networks/ResNet20"
))


if __name__ == "__main__":
    FashionMNIST_logger = getLogger("FashionMNIST")
    FashionMNIST_logger.info(INFO)
    
    resnet20 = ResNet20().to(device)
    resnet20_logger = getLogger("resnet20")

    resnet20_optimizers = get_optimizers(resnet20)

    # ResNet20
    resnet20_logger.info("ResNet20 on FashionMNIST\n")
    # warm-up epoch
    resnet20_logger.info("Warm-up")
    start_epoch = 0
    start_epoch = train_until(resnet20, resnet20_logger, 0.2,
                              resnet20_optimizers[0.01])
    resnet20_logger.info("learning_rate = 0.1")
    train_epoch_range(resnet20, resnet20_logger, start_epoch, 10,
                      resnet20_optimizers[0.1])
    resnet20_logger.info("learning_rate = 0.01")
    train_epoch_range(resnet20, resnet20_logger, 10, 15,
                      resnet20_optimizers[0.01])
    resnet20_logger.info("learning_rate = 0.001")
    train_epoch_range(resnet20, resnet20_logger, 15, 20,
                      resnet20_optimizers[0.001])
    # train_epoch_range(resnet20, resnet20_logger, 0, 200,
    #                   resnet20_optimizer)
    # testing
    test(resnet20, resnet20_logger)
    if not os.path.exists(resnet20_path):
        os.mkdir(resnet20_path)
    torch.save(
        resnet20,
        os.path.abspath(os.path.join(
            resnet20_path, "./ResNet20.resnet20"
        )),
    )

    pass
