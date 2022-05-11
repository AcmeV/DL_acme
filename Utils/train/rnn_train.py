import copy
import csv
import math
import time

import numpy as np
import tqdm

import torch
from d2l import torch as d2l
from torch import nn, optim
from torch.nn.utils import clip_grad

from Utils.AverageMeter import AverageMeter, accuracy, f1_score
from Utils.initial import init_model, init_dataset
from Utils.test.rnn_test import predict_time_machine, test_function_value, rnn_test, normalized_confusion_matrix


def grad_clipping(model, theta):
    if isinstance(model, nn.Module):
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        params = model.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def machine_func_train_epoch(model, train_loader, loss_func, optimizer, device):
    '''
        'TimeMachine' and 'FunctionValue' dataset train an epoch
    :param model:
    :param train_loader:
    :param loss_func:
    :param optimizer:
    :param device:
    :return:
    '''
    metric = AverageMeter()

    state = None
    for X, Y in train_loader:
        if X.shape[0] == train_loader.batch_size:
            if state is None:
                # Initialize `state` when either it is the first iteration or
                # using random sampling
                state = model.begin_state(batch_size=X.shape[0], device=device)
            else:
                if isinstance(model, nn.Module) and not isinstance(state, tuple):
                    # `state` is a tensor for `nn.GRU`
                    state.detach_()
                else:
                    # `state` is a tuple of tensors for `nn.LSTM` and
                    # for our custom scratch implementation
                    for s in state:
                        s.detach_()

            X, y = X.to(device), Y.to(device)

            y_hat, _ = model(X, state)
            loss = loss_func(y_hat, y)
            if isinstance(optimizer, torch.optim.Optimizer):
                optimizer.zero_grad()
                loss.backward()
                grad_clipping(model, 1)
                optimizer.step()
            else:
                loss.backward()
                grad_clipping(model, 1)
                optimizer(batch_size=1)
            metric.update(loss.item() , y.numel())
    return math.exp(metric.avg)

def HAR_train_epoch(model, train_loader, test_loader, loss_func, optimizer, device, bsz):
    '''
        'HAR' dataset train an epoch
    :param model:
    :param train_loader:
    :param test_loader:
    :param loss_func:
    :param optimizer:
    :param device:
    :param bsz: trainning batch size
    :return: train and test norm(accuracy, f1 sacore and loss)
    '''

    losses = AverageMeter()
    top1 = AverageMeter()
    f1_scores = AverageMeter()

    state = model.begin_state(batch_size=bsz, device=device)

    for inputs, targets in tqdm.tqdm(train_loader, ncols=50):
        if len(targets) == bsz:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, state)
            loss = loss_func(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            clip_grad.clip_grad_norm_(model.parameters(), 15)
            optimizer.step()

            prec1 = accuracy(outputs.data, targets, topk=(1,))
            f1score = f1_score(outputs.data, targets)
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1[0].item(), inputs.size(0))
            f1_scores.update(f1score, inputs.size(0))

    test_acc, test_f1_score, test_loss = rnn_test(model, test_loader, device, loss_func)
    return top1.avg, f1_scores.avg, losses.avg, test_acc, test_f1_score, test_loss


def rnn_train(args):
    ################################ System Init #####################################

    device = torch.device(f'cuda:0' if args.device != 'cpu'
                                               and torch.cuda.is_available() else "cpu")
    print(f'Use {device} \n')

    model_name = args.model

    print(f'model: {model_name} | dataset: {args.dataset} | lr: {args.lr}(decay: {args.lr_decay}) | device: {device}\n')

    log_path = f'{args.log_dir}/Log_{model_name}_{args.dataset}_lr-{args.lr}_decay-{args.lr_decay}_bsz-{args.train_bsz}_step-{args.num_step}.csv'
    model_pkl_path = f'{args.model_save_dir}/{model_name}_{args.dataset}_lr-{args.lr}_decay-{args.lr_decay}_bsz-{args.train_bsz}_step-{args.num_step}.pkl'
    matrix_path = f'{args.log_dir}/Matrix_{model_name}_{args.dataset}_lr-{args.lr}_decay-{args.lr_decay}_bsz-{args.train_bsz}_step-{args.num_step}.npy'

    log_file = open(log_path, "w", newline='')
    log_writer = csv.writer(log_file)
    if args.model == 'MyRNN' or args.model == 'RNN':
        log_writer.writerow(['Epoch', 'Perplexity', 'Time'])
    else:
        log_writer.writerow(['Epoch', 'TrainAcc', 'TrainLoss', 'TestAcc', 'TestLoss', 'Time'])
    log_file.flush()
    ##################################################################################

    ################################ Param Init ######################################
    model = init_model(args)

    train_loader, test_loader = init_dataset(args)
    if args.lr_decay == 1:
        lr_opt = lambda lr, epoch: lr * (0.1 ** (float(epoch) / 20))  # lr changes with epoch
    else:
        lr_opt = lambda lr, epoch: lr

    # Select Optimizer
    if isinstance(model, nn.Module):
        if args.optim == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
        else:
            optimizer = optim.SGD(model.parameters(), lr=args.lr)
    else:
        optimizer = lambda batch_size: d2l.sgd(model.parameters(), args.lr, batch_size)

    if args.dataset == 'TimeMachine':
        loss_func = nn.CrossEntropyLoss()  # CrossEntropy Loss Function
    elif args.dataset == 'FunctionValue':
        loss_func = nn.MSELoss()
    else:
        loss_func = nn.CrossEntropyLoss()

    model.to(device)
    ##################################################################################
    best_model = copy.deepcopy(model)
    for epoch in range(int(args.epochs)):
        best_acc = 0.

        lr_cur = lr_opt(args.lr, epoch)  # speed change

        if args.lr_decay == 1:
            if isinstance(model, nn.Module):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_cur
            else:
                optimizer = lambda batch_size: d2l.sgd(model.parameters(), lr_cur, batch_size)


        start_t = time.time()

        if args.dataset == 'TimeMachine' or args.dataset == 'FunctionValue':
            # 'TimeMachine' and 'FunctionValue' dataset use Perplexity as norm
            ppl = machine_func_train_epoch(model, train_loader, loss_func, optimizer, device)
            epoch_t = format(time.time() - start_t, '.3f')
            print('epoch: {epoch} | Perplexity: {ppl:.4f} | time:{epoch_t}s'.format(
                epoch=epoch, ppl=ppl, epoch_t=epoch_t))
            log_writer.writerow([epoch, format(ppl, '.4f'), epoch_t])
            log_file.flush()
        elif args.dataset == 'HAR':
            # 'HAR' dataset use accuracy, loss and f1 score as norm
            train_acc, train_f1_score, train_loss, test_acc, test_f1_score, test_loss = \
                HAR_train_epoch(model, train_loader, test_loader, loss_func, optimizer, device, args.train_bsz)

            if test_acc > best_acc:
                best_acc = test_acc
                best_model = copy.deepcopy(model)

            epoch_t = format(time.time() - start_t, '.3f')
            print('epoch: {epoch} | TrainAcc: {train_acc:.2f}% | TrainF1: {train_f1:.4f} | TrainLoss: {train_loss: .4f} | '
                  'TestAcc: {test_acc:.2f}% | TestF1: {test_f1:.4f} | TestLoss: {test_loss: .4f} | time:{epoch_t}s'.format(
                epoch=epoch, train_acc=train_acc, train_f1=train_f1_score, train_loss=train_loss,
                test_acc=test_acc, test_f1=test_f1_score, test_loss=test_loss, epoch_t=epoch_t))
            log_writer.writerow([epoch, format(train_acc, '.2f'), train_loss,
                                 format(test_acc, '.2f'), test_loss, epoch_t])
            log_file.flush()
            # save best model
            torch.save(best_model.state_dict(), model_pkl_path)

    if args.dataset == 'HAR':
        # save confusion matrix on test data
        matrix = normalized_confusion_matrix(best_model, test_loader)
        np.save(matrix_path, matrix)


    #     if args.dataset == 'TimeMachine':
    #         if epoch % 50 == 0:
    #             print(f'epoch{epoch}: ', end='')
    #             predict = lambda prefix: predict_time_machine(prefix, 50, model, test_loader, device)
    #             words = predict('time traveller ').split(' ')
    #             print(words[0], words[1], sep=' ', end= ' ')
    #             for i in range(2, len(words)):
    #                 print(f'\033[41m{words[i]} \033[0m', end='')
    #             print()
    #     else:
    #         print('epoch: {epoch} | Perplexity: {ppl:.4f} | time:{epoch_t}s'.format(
    #             epoch=epoch, ppl=ppl, epoch_t=epoch_t))
    #
    # if args.dataset == 'TimeMachine':
    #     predict = lambda prefix: predict_time_machine(prefix, 50, model, test_loader, device)
    #     words = predict('time traveller ').split(' ')
    #     print(words[0], words[1], sep=' ', end=' ')
    #     for i in range(2, len(words)):
    #         print(f'\033[41m{words[i]} \033[0m', end='')
    #     print()
    # elif args.dataset == 'FunctionValue':
    #     test_function_value(model, test_loader, device)
