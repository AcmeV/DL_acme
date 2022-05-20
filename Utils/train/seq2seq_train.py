import csv

import torch
from d2l import torch as d2l
from torch import nn, optim

from Utils import AverageMeter, init_model, init_dataset
from Utils.dl_utils import MaskedSoftmaxCELoss
from Utils.test.seq2seq_test import seq2seq_predict, seq2seq_test


def seq2seq_train_epoch(model, train_loader, tgt_vocab, optimizer, criterion, device):
    loss_meter = AverageMeter()

    for source, target in train_loader:
        optimizer.zero_grad()
        X, X_valid_len = source[0].to(device), source[1].to(device)
        Y, Y_valid_len = target[0].to(device), target[1].to(device)

        bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                           device=device).reshape(-1, 1)
        dec_input = d2l.concat([bos, Y[:, :-1]], 1)

        Y_hat, _ = model(X, dec_input, X_valid_len)
        loss = criterion(Y_hat, Y, Y_valid_len)
        loss.sum().backward()  # Make the loss scalar for `backward`

        d2l.grad_clipping(model, 1)

        num_tokens = Y_valid_len.sum()
        loss_meter.update(loss.mean().item(), num_tokens)

        optimizer.step()

    train_loss = format(loss_meter.avg, '.4f')

    return train_loss

def seq2seq_train(args):

    device = torch.device(f'cuda:0' if args.device != 'cpu'
                                       and torch.cuda.is_available() else "cpu")
    print(f'Use {device} \n')

    model_name = args.model

    print(f'model: {model_name} | dataset: {args.dataset} | lr: {args.lr}(decay: {args.lr_decay}) | device: {device}\n')

    log_path = f'{args.log_dir}/Log_{model_name}_{args.dataset}_lr-{args.lr}_decay-{args.lr_decay}_bsz-{args.train_bsz}.csv'

    log_file = open(log_path, "w", newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['Epoch', 'BLEU', 'Loss'])
    log_file.flush()
    ##################################################################################

    ################################ Param Init ######################################
    model = init_model(args)

    train_loader, test_loader, src_vocab, tgt_vocab = init_dataset(args)

    if args.lr_decay == 1:
        lr_opt = lambda lr, epoch: lr * (0.1 ** (float(epoch) / 20))  # lr changes with epoch
    else:
        lr_opt = lambda lr, epoch: lr

    # Select Optimizer
    if args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr)

    criterion = MaskedSoftmaxCELoss()

    model.to(device)
    ##################################################################################
    pred = None
    for epoch in range(args.epochs):
        model.train()
        if args.lr_decay == 1:
            lr_cur = lr_opt(args.lr, epoch)  # speed change
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_cur
        train_loss = seq2seq_train_epoch(model, train_loader, tgt_vocab, optimizer, criterion, device)

        bleu, pred, _ = seq2seq_test(model, test_loader, src_vocab, tgt_vocab, args.num_step, device, False)

        bleu = format(bleu, '.4f')

        print(f'Epoch: {epoch} | Loss: {train_loss} | BLEU: {bleu}')
        log_writer.writerow([epoch, bleu, train_loss])

    print('Eng\t\tFra\t\ttranslation')
    for i in range(len(pred[0])):
        print(f'{pred[0][i]}\t{pred[1][i]}\t{pred[2][i]}')
    print('==========================================================')