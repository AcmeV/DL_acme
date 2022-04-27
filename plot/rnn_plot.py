import matplotlib.pyplot as plt
import pandas as pd
from numpy import mean

plot_sep = 10
font = {
    'size': 13
}

def load_log(path):
    log = pd.read_csv(path)
    epochs, perplexities, times = [], [], []

    for i in range(len(log['Epoch'])):
    # for i in range(50):
        if i % plot_sep == 0:
            epochs.append(log['Epoch'][i])
            perplexities.append(float(log['Perplexity'][i]))
            times.append(float(log['Time'][i]))

    return epochs, perplexities, times

def ppl_plot(epochs, ppls, colors, labels, title):
    plt.figure(1, figsize=(6, 5))

    plt.title(title)
    plt.xlabel('Epoch', font)
    plt.ylabel('Perplexity', font)

    for i in range(len(epochs)):
        plt.plot(epochs[i], ppls[i], color=colors[i], label=labels[i])
        print(f'{labels[i]}: max acc = {max(ppls[i])}%, avg acc = {format(mean(ppls[i][30:]), ".2f")}%')
    plt.legend(loc='upper right', fontsize=12, ncol=2)
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

def plot_diff_step(model, dataset, lr):

    colors = [
        'green',
        'orange',
        'steelblue',
        'red'
    ]

    labels = [
        'time step = 10',
        'time step = 25',
        'time step = 35',
        'time step = 50',
    ]

    paths = [
        f'./logs/Log_{model}_{dataset}_lr-{lr}_decay-0_bsz-32_step-10.csv',
        f'./logs/Log_{model}_{dataset}_lr-{lr}_decay-0_bsz-32_step-25.csv',
        f'./logs/Log_{model}_{dataset}_lr-{lr}_decay-0_bsz-32_step-35.csv',
        f'./logs/Log_{model}_{dataset}_lr-{lr}_decay-0_bsz-32_step-50.csv',
    ]

    acc_title = f'Perplexity comparision of diffrent time step\n\nmodel: {model} | dataset: {dataset} | lr: {lr}'

    epochs, ppls = [], []
    for i in range(len(paths)):
        epoch, ppl, _ = load_log(paths[i])
        epochs.append(epoch)
        ppls.append(ppl)


    ppl_plot(epochs, ppls, colors, labels, acc_title)


def plot_if_random():
    colors = [
        'steelblue',
        'red'
    ]

    labels = [
        'use random',
        'no random',
    ]

    paths = [
        f'./logs/Log_RNN_FunctionValue_lr-0.001_decay-0_bsz-32_step-10_1.csv',
        f'./logs/Log_RNN_FunctionValue_lr-0.001_decay-0_bsz-32_step-10_2.csv',

    ]

    acc_title = f'Perplexity comparision of diffrent function prediction\n\nmodel: RNN | lr: 0.001'

    epochs, ppls = [], []
    for i in range(len(paths)):
        epoch, ppl, _ = load_log(paths[i])
        epochs.append(epoch)
        ppls.append(ppl)

    ppl_plot(epochs, ppls, colors, labels, acc_title)

if __name__ == '__main__':
    plot_sep = 1
    model = 'RNN'
    dataset = 'TimeMachine'
    lr = 1

    plot_diff_step(model, dataset, lr)
    # plot_if_random()