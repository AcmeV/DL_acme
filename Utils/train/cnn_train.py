import csv
import os
import time

import torch
import tqdm
from torch import optim, nn
from torch.autograd import Variable

from Utils import AverageMeter
from Utils.initial import init_model, init_dataset
from Utils.test.cnn_test import cnn_test
from Utils.AverageMeter import accuracy


def cnn_train(args):
    ################################ System Init #####################################
    if args.device == 'gpus':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        gpus = [int(idx) for idx in list(args.gpus.split(','))]
        print(f'Use gpus: {gpus} \n')
    else:
        gpus = [0]
    device = torch.device(f'cuda:{gpus[0]}' if args.device != 'cpu'
                                               and torch.cuda.is_available() else "cpu")
    model_name = args.model
    if 'ResNet' in args.model or args.model == 'VGG':
        model_name = f'{args.model}{args.model_version}'

    print(f'model: {model_name} | dataset: {args.dataset} | lr: {args.lr}(decay: {args.lr_decay}) | device: {device}\n')

    log_file = open(f'{args.log_dir}/Log_{model_name}_{args.dataset}_lr-{args.lr}_decay-{args.lr_decay}.csv', "w",
                    newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['Epoch', 'TrainAcc', 'TrainLoss', 'TestAcc', 'TestLoss', 'Time'])
    log_file.flush()
    ##################################################################################

    ################################ Param Init ######################################
    model = init_model(args)
    # model = torchvision.models.resnet18(pretrained=True)

    train_data, test_data = init_dataset(args)
    if args.lr_decay == 1:
        lr_opt = lambda lr, epoch: lr * (0.1 ** (float(epoch) / 20))  # lr changes with epoch
    else:
        lr_opt = lambda lr, epoch: lr
    # Select Optimizer
    if args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    loss_func = nn.CrossEntropyLoss()  # CrossEntropy Loss Function

    if args.device != 'cpu' and torch.cuda.is_available():  # parallel trainning mode
        model = model.cuda()
        model = torch.nn.DataParallel(model, device_ids=gpus)
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

        for inputs, targets in tqdm.tqdm(train_data, ncols=50):
            inputs, targets = Variable(inputs).to(device), Variable(targets).to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prec1 = accuracy(outputs.data, targets, topk=(1,))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1[0].item(), inputs.size(0))

        ############### norm for trainning set ###########################
        train_loss = losses.avg
        train_acc = top1.avg

        ############## norm for test set #################################
        test_acc, test_loss = cnn_test(model, test_data, device, loss_func)
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