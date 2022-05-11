import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn import metrics

from Utils.AverageMeter import AverageMeter, accuracy, f1_score


def predict_time_machine(prefix, num_preds, rnn_net, vocab, device):
    outputs = [vocab[prefix[0]]]
    state = rnn_net.begin_state(batch_size=1, device=device)
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1,1))
    for y in prefix[1:]:
        _, state = rnn_net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):
        y, state = rnn_net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


def test_function_value(model, test_loader, device):
    preds = []
    labels = []
    model.to(device)
    with torch.no_grad():
        state = None
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred, _ = model(x, state)
            preds.extend(pred.data.squeeze(1).tolist())
            labels.extend(y.data.squeeze(1).tolist())

    plt.plot([ele for ele in preds[0:100]], "r", label="pred")
    plt.plot([ele for ele in labels[0:100]], "b", label="real")
    plt.legend(loc='lower right', fontsize=12, ncol=2)
    plt.show()

def rnn_test(model, test_loader, device, loss_func):
    losses = AverageMeter()
    top1 = AverageMeter()
    f1_scores = AverageMeter()

    model.to(device)
    loss_func.to(device)
    bsz = test_loader.batch_size
    with torch.no_grad():
        state = model.begin_state(batch_size=bsz, device=device)
        for inputs, targets in test_loader:
            if len(targets) == bsz:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs, state)
                loss = loss_func(outputs, targets)

                prec1 = accuracy(outputs.data, targets, topk=(1,))
                f1score = f1_score(outputs.data, targets)
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1[0].item(), inputs.size(0))
                f1_scores.update(f1score, inputs.size(0))

    test_acc = top1.avg
    test_loss = losses.avg
    test_f1_score = f1_scores.avg

    return test_acc, test_f1_score, test_loss

def normalized_confusion_matrix(model, test_loader):
    device = torch.device('cpu')
    model.to(device)
    with torch.no_grad():
        inputs, targets = [], []
        for input, target in test_loader:
            inputs.append(input)
            targets.append(target)
        inputs = torch.cat(inputs, dim=0)
        targets = torch.cat(targets, dim=0)
        state = model.begin_state(batch_size=len(targets), device=device)
        outputs = model(inputs, state)

        _, top_class = outputs.topk(1, dim=1)

        confusion_matrix = metrics.confusion_matrix(top_class, targets)
        normalized_confusion_matrix = np.array(confusion_matrix, dtype=np.float32) / np.sum(confusion_matrix) * 100


    return normalized_confusion_matrix