import os
import sys
import torch
pass
base_folder = os.path.dirname(os.path.abspath(__file__))
if base_folder not in sys.path:
    sys.path.append(base_folder)
if True:
    from Networks.Networks import ResNet110, NonResNet110, ResNet32, NonResNet32
    from Networks.Training import get_optimizers, train_epoch_range, train_until, test
    from Data.CIFAR10 import CIFAR10_info
    from Log.Logger import getLogger


device = torch.device("cuda") if torch.cuda.is_available()\
    else torch.device("cpu")
criterion = torch.nn.CrossEntropyLoss()


resnet110_path = os.path.abspath(os.path.join(
    base_folder, "./Networks/ResNet110"
))
nonresnet110_path = os.path.abspath(os.path.join(
    base_folder, "./Networks/NonResNet110"
))
resnet32_path = os.path.abspath(os.path.join(
    base_folder, "./Networks/ResNet32"
))
nonresnet32_path = os.path.abspath(os.path.join(
    base_folder, "./Networks/NonResNet32"
))


if __name__ == "__main__":
    # CIFAR10_logger = getLogger("CIFAR10 for 110-layer models")
    # CIFAR10_logger = getLogger("CIFAR10 for 32-layer models")
    CIFAR10_logger = getLogger("CIFAR10")
    CIFAR10_logger.info(CIFAR10_info)

    resnet110 = ResNet110().to(device)
    resnet110_logger = getLogger("resnet110")

    nonresnet110 = NonResNet110().to(device)
    nonresnet110_logger = getLogger("nonresnet110")

    resnet110_optimizers = get_optimizers(resnet110)
    nonresnet110_optimizers = get_optimizers(nonresnet110)

    # ResNet110
    resnet110_logger.info("ResNet110 on CIFAR-10\n")
    # warm-up epoch
    resnet110_logger.info("Warm-up")
    start_epoch = 0
    start_epoch = train_until(resnet110, resnet110_logger, 0.2,
                              resnet110_optimizers[0.01])
    resnet110_logger.info("learning_rate = 0.1")
    train_epoch_range(resnet110, resnet110_logger, start_epoch, 100,
                      resnet110_optimizers[0.1])
    resnet110_logger.info("learning_rate = 0.01")
    train_epoch_range(resnet110, resnet110_logger, 100, 150,
                      resnet110_optimizers[0.01])
    resnet110_logger.info("learning_rate = 0.001")
    train_epoch_range(resnet110, resnet110_logger, 150, 200,
                      resnet110_optimizers[0.001])
    # train_epoch_range(resnet110, resnet110_logger, 0, 200,
    #                   resnet110_optimizer)
    # testing
    test(resnet110, resnet110_logger)
    if not os.path.exists(resnet110_path):
        os.mkdir(resnet110_path)
    torch.save(
        resnet110,
        os.path.abspath(os.path.join(
            resnet110_path, "./ResNet110.resnet110"
        )),
    )

    # NonResNet110
    nonresnet110_logger.info("NonResNet110 on CIFAR-10\n")
    # warm-up epoch
    nonresnet110_logger.info("Warm-up")
    start_epoch = 0
    start_epoch = train_until(nonresnet110, nonresnet110_logger, 0.2,
                              nonresnet110_optimizers[0.01])
    nonresnet110_logger.info("learning_rate = 0.1")
    train_epoch_range(nonresnet110, nonresnet110_logger, start_epoch, 100,
                      nonresnet110_optimizers[0.1])
    nonresnet110_logger.info("learning_rate = 0.01")
    train_epoch_range(nonresnet110, nonresnet110_logger, 100, 150,
                      nonresnet110_optimizers[0.01])
    nonresnet110_logger.info("learning_rate = 0.001")
    train_epoch_range(nonresnet110, nonresnet110_logger, 150, 200,
                      nonresnet110_optimizers[0.001])
    # train_epoch_range(nonresnet110, nonresnet110_logger, 0, 200,
    #                   nonresnet110_optimizer)
    # testing
    test(nonresnet110, nonresnet110_logger)
    if not os.path.exists(nonresnet110_path):
        os.mkdir(nonresnet110_path)
    torch.save(
        nonresnet110,
        os.path.abspath(os.path.join(
            nonresnet110_path, "./NonResNet110.nonresnet110"
        )),
    )
    
    resnet32 = ResNet32().to(device)
    resnet32_logger = getLogger("resnet32")

    nonresnet32 = NonResNet32().to(device)
    nonresnet32_logger = getLogger("nonresnet32")

    resnet32_optimizers = get_optimizers(resnet32)
    nonresnet32_optimizers = get_optimizers(nonresnet32)

    # ResNet32
    resnet32_logger.info("ResNet32 on CIFAR-10\n")
    # warm-up epoch
    resnet32_logger.info("Warm-up")
    start_epoch = 0
    start_epoch = train_until(resnet32, resnet32_logger, 0.2,
                              resnet32_optimizers[0.01])
    resnet32_logger.info("learning_rate = 0.1")
    train_epoch_range(resnet32, resnet32_logger, start_epoch, 100,
                      resnet32_optimizers[0.1])
    resnet32_logger.info("learning_rate = 0.01")
    train_epoch_range(resnet32, resnet32_logger, 100, 150,
                      resnet32_optimizers[0.01])
    resnet32_logger.info("learning_rate = 0.001")
    train_epoch_range(resnet32, resnet32_logger, 150, 200,
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

    # NonResNet32
    nonresnet32_logger.info("NonResNet32 on CIFAR-10\n")
    # warm-up epoch
    nonresnet32_logger.info("Warm-up")
    start_epoch = 0
    start_epoch = train_until(nonresnet32, nonresnet32_logger, 0.2,
                              nonresnet32_optimizers[0.01])
    nonresnet32_logger.info("learning_rate = 0.1")
    train_epoch_range(nonresnet32, nonresnet32_logger, start_epoch, 100,
                      nonresnet32_optimizers[0.1])
    nonresnet32_logger.info("learning_rate = 0.01")
    train_epoch_range(nonresnet32, nonresnet32_logger, 100, 150,
                      nonresnet32_optimizers[0.01])
    nonresnet32_logger.info("learning_rate = 0.001")
    train_epoch_range(nonresnet32, nonresnet32_logger, 150, 200,
                      nonresnet32_optimizers[0.001])
    # train_epoch_range(nonresnet32, nonresnet32_logger, 0, 200,
    #                   nonresnet32_optimizer)
    # testing
    test(nonresnet32, nonresnet32_logger)
    if not os.path.exists(nonresnet32_path):
        os.mkdir(nonresnet32_path)
    torch.save(
        nonresnet32,
        os.path.abspath(os.path.join(
            nonresnet32_path, "./NonResNet32.nonresnet32"
        )),
    )

    pass
