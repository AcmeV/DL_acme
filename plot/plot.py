import matplotlib.pyplot as plt
import pandas as pd
from numpy import mean


font = {
    'size': 13
}

def load_log(path):
    log = pd.read_csv(path)
    epochs, accs, losses = [], [], []

    for i in range(len(log['Epoch'])):
        epochs.append(log['Epoch'][i])
        accs.append(log['Acc'][i])
        losses.append(log['Loss'][i])

    return epochs, accs, losses

def acc_plot(epochs, accs, colors, labels, title):
    plt.figure(1, figsize=(6, 5))

    plt.title(title)
    plt.xlabel('Epoch', font)
    plt.ylabel('Accuracy(%)', font)

    for i in range(len(epochs)):
        plt.plot(epochs[i], accs[i], color=colors[i], label=labels[i])
        print(f'{labels[i]}: max acc = {max(accs[i])}%, avg acc = {format(mean(accs[i][30:]), ".2f")}%')
    plt.legend(loc='lower right', fontsize=12, ncol=2)
    plt.show()

def loss_plot(epochs, losses, colors, labels, title):
    plt.figure(1, figsize=(6, 5))

    plt.title(title)
    plt.xlabel('Epoch', font)
    plt.ylabel('Loss', font)
    for i in range(len(epochs)):
        plt.plot(epochs[i], losses[i], color=colors[i], label=labels[i])
        print(f'{labels[i]}: min loss = {min(losses[i])}, avg loss = {format(mean(losses[i][30:]), ".2f")}')

    plt.legend(loc='upper right', fontsize=12, ncol=2)
    plt.show()

def plot_diff_lr():
    model = 'CNN'
    dataset = 'mnist'

    colors = ['green', 'steelblue', 'red']
    labels = ['lr = 0.1', 'lr = 0.01', 'lr = 0.001']
    paths = [
        f'./logs/Log_{model}_{dataset}_lr-0.1.csv',
        f'./logs/Log_{model}_{dataset}_lr-0.01.csv',
        f'./logs/Log_{model}_{dataset}_lr-0.001.csv'
    ]
    acc_title = 'Accuracy comparision of diffrent lr\nmodel: CNN | dataset: MNIST'
    loss_title = 'Loss comparision of diffrent lr\nmodel: CNN | dataset: MNIST'

    epochs, accs, losses = [], [], []
    for i in range(3):
        epoch, acc, loss = load_log(paths[i])
        epochs.append(epoch)
        accs.append(acc)
        losses.append(loss)

    acc_plot(epochs, accs, colors, labels, acc_title)
    loss_plot(epochs, losses, colors, labels, loss_title)

def plot_diff_model():
    acc_title = 'Accuracy comparision of diffrent model\nlr: 0.01 | dataset: MNIST'
    loss_title = 'Loss comparision of diffrent model\nlr: 0.01 | dataset: MNIST'

    dataset = 'mnist'

    colors = ['green', 'red']
    labels = ['ANN', 'CNN']
    paths = [
        f'./logs/Log_ANN_{dataset}_lr-0.01.csv',
        f'./logs/Log_CNN_{dataset}_lr-0.01.csv'
    ]

    epochs, accs, losses = [], [], []
    for i in range(len(paths)):
        epoch, acc, loss = load_log(paths[i])
        epochs.append(epoch)
        accs.append(acc)
        losses.append(loss)

    acc_plot(epochs, accs, colors, labels, acc_title)
    loss_plot(epochs, losses, colors, labels, loss_title)


if __name__ == '__main__':
    plot_diff_lr()
    # plot_diff_model()