import os
import sys
import torch
from tqdm import tqdm, trange
from logging import Logger
from typing import Iterable
base_folder = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
if base_folder not in sys.path:
    sys.path.append(base_folder)
if True:
    from config import device
    from Networks import GoogLeNet

criterion = torch.nn.CrossEntropyLoss()


def get_optimizers(
    model: torch.nn.Module,
    learning_rate: Iterable[float] = (0.1, 0.01, 0.001)
):
    return {
        lr: torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=0.0001,
        )
        for lr in learning_rate
    }


def train_epoch(
    model: torch.nn.Module,
    data,
    logger: Logger,
    epoch: int,
    optimizer: torch.optim.Optimizer,
) -> tuple[float]:
    message = "[loss]\ttrain:{:.3f},\tval:{:.3f}\n[acc]\ttrain:{:.3f},\tval:{:.3f}\n"

    logger.info("\n[Epoch {:0>4d}]".format(epoch+1))
    train_loss, val_loss, train_acc, val_acc = 0, 0, 0, 0

    model.train()
    print("\nTraining:")
    for sample in tqdm(data.train_loader):
        x, y = sample
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        h = model(x)
        if isinstance(model, GoogLeNet):
            loss = 0
            for i in range(len(h)):
                loss += criterion(h[i], y) * model.training_weights[i]
            h = h[0]
        else:
            loss = criterion(h, y)
        loss.backward()
        optimizer.step()

        h = torch.argmax(h, dim=1)
        acc = torch.sum((h == y).to(torch.int32))

        train_loss += len(y)*(float(loss))
        train_acc += int(acc)
    train_loss /= data.len_train
    train_acc /= data.len_train

    model.eval()
    print("\nValidating:")
    with torch.no_grad():
        for sample in tqdm(data.val_loader):
            x, y = sample
            x, y = x.to(device), y.to(device)

            h = model(x)
            loss = criterion(h, y)

            h = torch.argmax(h, dim=1)
            acc = torch.sum((h == y).to(torch.int32))

            val_loss += len(y)*(float(loss))
            val_acc += int(acc)
        val_loss /= data.len_val
        val_acc /= data.len_val

    result = train_loss, val_loss, train_acc, val_acc
    print("")
    logger.info(message.format(*result))
    return result


def train_epoch_range(
    model: torch.nn.Module,
    data,
    logger: Logger,
    start: int,
    stop: int,
    optimizer: torch.optim.Optimizer,
) -> None:
    for epoch in trange(start, stop):
        train_epoch(model, data, logger, epoch, optimizer)


def train_until(
    model: torch.nn.Module,
    data,
    logger: Logger,
    threshold: float,
    optimizer: torch.optim.Optimizer,
) -> int:
    epoch = 0
    train_loss, val_loss, train_acc, val_acc = 0, 0, 0, 0
    while train_acc <= threshold:
        train_loss, val_loss, train_acc, val_acc = \
            train_epoch(model, data, logger, epoch, optimizer)
        epoch += 1
    return epoch


@torch.no_grad()
def test(
    model: torch.nn.Module,
    data,
    logger: Logger,
) -> None:
    logger.info("\nTesting: ")

    message = "[loss]\ttest:{:.3f}\n[acc]\ttest:{:.3f}\n"
    test_loss, test_acc = 0, 0

    model.eval()
    print("\nTesting:")
    for sample in tqdm(data.test_loader):
        x, y = sample
        x, y = x.to(device), y.to(device)

        h = model(x)
        loss = criterion(h, y)

        h = torch.argmax(h, dim=1)
        acc = torch.sum(h == y)

        test_loss += len(y)*(float(loss))
        test_acc += int(acc)
    test_loss /= data.len_test
    test_acc /= data.len_test

    print("")
    logger.info(message.format(test_loss, test_acc))


def check(
    model: torch.nn.Module,
    data,
    logger: Logger,
) -> None:
    logger.info("\nChecking: ")

    message = "[loss]\tcheck:{:.3f}\n[acc]\tcheck:{:.3f}\n"
    check_loss, check_acc = 0, 0
    examples = torch.zeros(
        (10,), dtype=torch.float32, device=device)
    confusion_matrix = torch.zeros(
        (10, 10), dtype=torch.float32, device=device)

    model.eval()
    print("\nChecking:")
    for sample in tqdm(data.check_loader):
        x, y = sample
        x, y = x.to(device), y.to(device)

        h = model(x)
        loss = criterion(h, y)

        h = torch.argmax(h, dim=1)
        acc = torch.sum(h == y)

        check_loss += len(y)*(float(loss))
        check_acc += int(acc)

        for index in zip(y, h):
            confusion_matrix[index] += 1
            examples[index[0]] += 1
    check_loss /= data.len_test
    check_acc /= data.len_test
    confusion_matrix = (confusion_matrix.T / examples).T

    print("")
    logger.info(message.format(check_loss, check_acc))
    cm = confusion_matrix.cpu().numpy()
    logger.info("[confusion matrix]")
    logger.info(
        "\n".join(["\t".join(["{:.3f}".format(j) for j in i]) for i in cm]))
    return cm
