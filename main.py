import csv
import os
import time
import argparse

import torch
import torchvision
from torch import optim, nn
from torch.autograd import Variable

from Utils.test import test_model, AverageMeter, accuracy
from Utils.initial import init_model, init_dataset

parser = argparse.ArgumentParser()
# System settings
parser.add_argument('--data-dir', type=str, default='./data/')
parser.add_argument('--model-package', type=str, default='models')
parser.add_argument('--log-dir', type=str, default='./logs')
parser.add_argument('--device', type=str, default='cpu',
                    choices=('cpu', 'gpu', 'gpus'))
parser.add_argument('--gpus', type=str, default='0,1,2,3',
                    help='ID of GPUs to use, eg. 1,3')

# Model
parser.add_argument('--model', type=str, default='CNN',
                    choices=('NN', 'CNN', 'AlexNet', 'NiN', 'VGG',
                             'GoogLeNet', 'ResNet', 'SEResNet'))
parser.add_argument('--model-version', type=int, default=11,
                    choices=(18, 34, 50, 101, 152,
                             11, 13, 16, 19),
                    help='version for ResNet or VGG,'
                         '(18, 34, 50, 101, 152) is for ResNet,'
                         '(11, 13, 16, 19) is for VGG')
# Dataset
parser.add_argument('--dataset', type=str, default='CIFAR10',
                    choices=('MNIST', 'CIFAR10', 'TinyImageNet'))
parser.add_argument('--train-bsz', type=int, default=128)
parser.add_argument('--test-bsz', type=int, default=128)

# Hyper-parameters
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr-decay', type=int, default=0,
                    choices=(0, 1))
parser.add_argument('--optim', type=str, default='SGD',
                    choices=('SGD', 'Adam'))
args = parser.parse_args()

if __name__ == '__main__':
    ################################ System Init #####################################
    if args.device == 'gpus':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        gpus = [int(idx) for idx in list(args.gpus.split(','))]
        print(f'Use gpus: {gpus} \n')
    else:
        gpus = [0]
    device = torch.device(f'cuda:{gpus[0]}' if args.device != 'cpu'
                    and torch.cuda.is_available() else "cpu")

    if args.model != 'ResNet' and args.model != 'SEResNet' and args.model != 'VGG':
        print(f'model: {args.model} | dataset: {args.dataset} | lr: {args.lr}(decay: {args.lr_decay}) | device: {device}\n')

        log_file = open(f'{args.log_dir}/Log_{args.model}_{args.dataset}_lr-{args.lr}_decay-{args.lr_decay}.csv',
                        "w", newline='')
    else:
        print(f'model: {args.model}-{args.model_version} | dataset: {args.dataset} | lr: {args.lr}(decay: {args.lr_decay}) | device: {device}\n')

        log_file = open(f'{args.log_dir}/Log_{args.model}{args.model_version}_{args.dataset}_lr-{args.lr}_decay-{args.lr_decay}.csv',
                        "w", newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['Epoch', 'TrainAcc', 'TrainLoss', 'TestAcc', 'TestLoss', 'Time'])
    log_file.flush()
    ##################################################################################

    ################################ Param Init ######################################
    model = init_model(args)

    # model = torchvision.models.resnet18(pretrained=True)

    train_data, test_data = init_dataset(args)
    train_len = len(train_data.dataset)
    test_len = len(test_data.dataset)
    if args.lr_decay ==  1:
        lr_opt = lambda lr, epoch: lr * (0.1 ** (float(epoch) / 20))  # lr changes with epoch
    else:
        lr_opt = lambda lr, epoch: lr
    # Select Optimizer
    if args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    loss_func = nn.CrossEntropyLoss() # CrossEntropy Loss Function

    if args.device != 'cpu': # parallel trainning mode
        model = model.cuda()
        model = torch.nn.DataParallel(model, device_ids=gpus)

    # correct, acc, test_loss = test_model(model, test_data, device, loss_func)
    # test_len = len(test_data.dataset)
    # print(f'Init model test |  test loss: {test_loss} | acc: {acc}%({correct} / {test_len})\n')
    ##################################################################################

    ################################ Trainning Section ###############################
    print("Trainning Start\n")
    model.train()
    for epoch in range(args.epochs):

        losses = AverageMeter()
        top1 = AverageMeter()

        lr_cur = lr_opt(args.lr, epoch)  # speed change
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_cur

        start_t = time.time()

        for batch_idx, (inputs, targets) in enumerate(train_data):
            inputs, targets = Variable(inputs).to(device), Variable(targets).to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prec1 = accuracy(outputs.data, targets, topk=(1, ))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1[0].item(), inputs.size(0))

            if batch_idx % 200 == 0:
                print(f'epoch: {epoch} | batch: {batch_idx}')

        ############### norm for trainning set ###########################
        train_loss = losses.val
        train_acc = top1.avg

        ############## norm for test set #################################
        test_acc, test_loss = test_model(model, test_data, device, loss_func)
        epoch_t = format(time.time() - start_t, '.3f')

        ######################### record #################################
        log_writer.writerow([epoch, train_acc, train_loss, test_acc, test_loss, epoch_t])
        log_file.flush()
        print('epoch: {epoch} | loss: {train_loss:.4f} |  acc: {train_acc:.2f}% | '
              'test loss: {test_loss:.4f} | test acc: {test_acc:.2f}% | time:{epoch_t}s'.format(
            epoch=epoch, train_loss=train_loss, train_acc=train_acc,
            test_loss=test_loss, test_acc=test_acc, epoch_t=epoch_t
        ))

    #####################################################################################

    log_file.close()
    print("System Stop !!")
