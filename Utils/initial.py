import csv
import os
import random

import torch
import models
import torchvision
import numpy as np

from Utils.dataset.FunctionValue import FunctionValue, gen_function_value
from Utils.dataset.TimeMachine import load_data_time_machine
from config.config import config
from Utils.dataset.TinyImageNet import TinyImageNet


def init_model(args):
    config_dict = config(args)
    if args.model == 'ResNet' or args.model == 'SEResNet' or args.model == 'VGG':
        model_name = f'{args.model}{args.model_version}'
        return eval(f'{config_dict[model_name][args.dataset]}()')
    else:
        return eval(f'{config_dict[args.model][args.dataset]}()')

def init_dataset(args):
    train_data, test_data = None, None
    if args.dataset == 'MNIST':
        train_data = torchvision.datasets.MNIST(root=f'{args.data_dir}', train=True,
                                                download=True, transform=torchvision.transforms.ToTensor())
        test_data = torchvision.datasets.MNIST(root=f'{args.data_dir}', train=False,
                                               download=True, transform=torchvision.transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.train_bsz, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bsz, shuffle=False)

        return train_loader, test_loader
    elif args.dataset == 'CIFAR10':
        train_data = torchvision.datasets.CIFAR10(root=f'{args.data_dir}', train=True,
                                                download=True, transform=torchvision.transforms.ToTensor())
        test_data = torchvision.datasets.CIFAR10(root=f'{args.data_dir}', train=False,
                                               download=True, transform=torchvision.transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.train_bsz, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bsz, shuffle=False)

        return train_loader, test_loader
    elif args.dataset == 'TinyImageNet':
        # 模拟 3 * 224 * 224
        # transforms_train = torchvision.transforms.transforms.Compose([
        #     torchvision.transforms.transforms.Resize((224, 224)),
        #     torchvision.transforms.transforms.RandomHorizontalFlip(),
        #     torchvision.transforms.transforms.ToTensor(),
        #     torchvision.transforms.transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        #     torchvision.transforms.transforms.RandomErasing(p=0.5, scale=(0.06, 0.08), ratio=(1, 3), value=0, inplace=True)
        # ])
        #
        # transforms_val = torchvision.transforms.transforms.Compose([
        #     torchvision.transforms.transforms.Resize((224, 224)),
        #     torchvision.transforms.transforms.ToTensor(),
        #     torchvision.transforms.transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
        # ])

        # 自定义Dataset
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        train_data = TinyImageNet(f'{args.data_dir}', transform=transform, train=True)
        test_data = TinyImageNet(f'{args.data_dir}', transform=transform, train=False)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.train_bsz, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bsz, shuffle=False)

        return train_loader, test_loader
    elif args.dataset == 'FunctionValue':
        if not os.path.exists(f'{args.data_dir}/FunctionValue.csv'):
            gen_function_value(f'{args.data_dir}/FunctionValue.csv')
        train_data = FunctionValue(dir=f'{args.data_dir}/FunctionValue.csv', train=True, time_step=args.num_step, transform=torchvision.transforms.ToTensor())
        test_data = FunctionValue(dir=f'{args.data_dir}/FunctionValue.csv', train=False, time_step=args.num_step, transform=torchvision.transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.train_bsz, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bsz, shuffle=False)

        return train_loader, test_loader
    elif args.dataset == 'TimeMachine':
        batch_size, num_step = int(args.train_bsz), int(args.num_step)
        train_iter, vocab = load_data_time_machine(batch_size, num_step, args.data_dir)
        return train_iter, vocab