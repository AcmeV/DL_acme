import argparse
import csv
import time

import torch
from torch import optim, nn

from Utils.initial import init_model, init_dataset
from Utils.test import test_model

parser = argparse.ArgumentParser()
# Model and Dataset
parser.add_argument('--model', type=str, default='ANN',
                    choices=('CNN', 'ANN'))
parser.add_argument('--dataset', type=str, default='mnist',
                    choices=('mnist'))
parser.add_argument('--train-bsz', type=int, default=128)
parser.add_argument('--test-bsz', type=int, default=128)

# Hyper-parameters
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--device', type=str, default='cpu',
                    choices=('cpu', 'gpu'))
parser.add_argument('--log-dir', type=str, default='./logs')
args = parser.parse_args()

if __name__ == '__main__':
    ######################### Init ##############################
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'gpu' else "cpu")

    log_file = open(f'{args.log_dir}/Log_{args.model}_{args.dataset}_lr-{args.lr}.csv',
                     "w", newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['Epoch', 'Acc', 'Loss'])
    #############################################################

    model = init_model(args).to(device)
    train_data, test_data = init_dataset(args)

    # Select SGD Optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # CrossEntropy Loss Function
    loss_func = nn.CrossEntropyLoss().to(device)

    correct, acc, test_loss = test_model(model, test_data, device, loss_func)
    test_len = len(test_data.dataset)
    print(f'Init model test |  test loss: {test_loss} | acc: {acc}%({correct} / {test_len})\n')

    ################################ trainning section #############################
    model.train()
    print("Start train \n")
    for epoch in range(args.epochs):

        start_t = time.time()

        for batch_idx, (inputs, targets) in enumerate(train_data):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, targets)
            loss.backward()
            optimizer.step()

        correct, acc, test_loss = test_model(model, test_data, device, loss_func)
        test_len = len(test_data.dataset)
        epoch_t = format(time.time() - start_t, '.3f')
        log_writer.writerow([epoch, acc, test_loss])
        log_file.flush()
        print(f'epoch: {epoch} |  test loss: {test_loss} | acc: {acc}%({correct} / {test_len}) | time:{epoch_t}s')
    ###################################################################################
    log_file.close()
    print("Stop !!")
