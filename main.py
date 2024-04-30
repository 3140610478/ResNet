import os
import sys
import torch
pass
base_folder = os.path.dirname(os.path.abspath(__file__))
if base_folder not in sys.path:
    sys.path.append(base_folder)
if True:
    from Networks.Networks import ResNet32
    from Networks.Training import get_optimizers, train_epoch_range, train_until, test
    from Data.FashionMNIST import INFO
    from Log.Logger import getLogger
    from config import device
    
criterion = torch.nn.CrossEntropyLoss()


resnet32_path = os.path.abspath(os.path.join(
    base_folder, "./Networks/ResNet32"
))


if __name__ == "__main__":
    FashionMNIST_logger = getLogger("FashionMNIST")
    FashionMNIST_logger.info(INFO)
    
    resnet32 = ResNet32().to(device)
    resnet32_logger = getLogger("resnet32")

    resnet32_optimizers = get_optimizers(resnet32)

    # ResNet32
    resnet32_logger.info("ResNet32 on FashionMNIST\n")
    # warm-up epoch
    resnet32_logger.info("Warm-up")
    start_epoch = 0
    start_epoch = train_until(resnet32, resnet32_logger, 0.2,
                              resnet32_optimizers[0.01])
    resnet32_logger.info("learning_rate = 0.1")
    train_epoch_range(resnet32, resnet32_logger, start_epoch, 10,
                      resnet32_optimizers[0.1])
    resnet32_logger.info("learning_rate = 0.01")
    train_epoch_range(resnet32, resnet32_logger, 10, 15,
                      resnet32_optimizers[0.01])
    resnet32_logger.info("learning_rate = 0.001")
    train_epoch_range(resnet32, resnet32_logger, 15, 20,
                      resnet32_optimizers[0.001])
    # train_epoch_range(resnet32, resnet32_logger, 0, 200,
    #                   resnet32_optimizer)
    # testing
    test(resnet32, resnet32_logger)
    if not os.path.exists(resnet32_path):
        os.mkdir(resnet32_path)
    torch.save(
        resnet32,
        os.path.abspath(os.path.join(
            resnet32_path, "./ResNet32.resnet32"
        )),
    )

    pass
