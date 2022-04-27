import torch

from Utils.AverageMeter import AverageMeter


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

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        a = correct[:k]
        b = a.view(-1)
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res