import torch


def test_model(model, test_data, device, loss_func):
    correct, test_loss = 0, 0.
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(test_data):
            data = data.to(device)
            targets = targets.to(device)
            output = model(data)
            test_loss += loss_func(output, targets).data.item()
            # get the index of the max log-probability
            pred = output.data.max(1)[1]
            correct += pred.eq(targets.data).sum().item()
    test_loss /= len(test_data)
    test_loss = format(test_loss, '.4f')
    acc = format(correct * 100 / len(test_data.dataset), '.2f')

    return correct, acc, test_loss