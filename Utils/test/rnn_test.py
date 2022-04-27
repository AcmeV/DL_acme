import torch
import matplotlib.pyplot as plt

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
            if x.shape[0] == test_loader.batch_size:
                if state is None:
                    state = model.begin_state(batch_size=x.shape[0], device=device)
                x, y = x.to(device), y.to(device)
                pred, _ = model(x, state)
                preds.extend(pred.data.squeeze(1).tolist())
                labels.extend(y.data.squeeze(1).tolist())

    plt.plot([ele for ele in preds[0:100]], "r", label="pred")
    plt.plot([ele for ele in labels[0:100]], "b", label="real")
    plt.legend(loc='lower right', fontsize=12, ncol=2)
    plt.show()