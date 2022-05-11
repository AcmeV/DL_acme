import torch

from Utils.AverageMeter import AverageMeter, accuracy


def cnn_test(model, test_loader, device, loss_func):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.to(device)
    loss_func.to(device)
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, targets)

            prec1 = accuracy(outputs.data, targets, topk=(1,))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1[0].item(), inputs.size(0))

    acc = top1.avg
    test_loss = losses.avg

    return acc, test_loss