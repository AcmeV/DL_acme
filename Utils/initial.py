import json
import os

import torch
import models
import torchvision

from Utils.dataset.FunctionValue import FunctionValue, gen_function_value
from Utils.dataset.HAR import HAR
from Utils.dataset.TimeMachine import load_data_time_machine
from Utils.dataset.TinyImageNet import TinyImageNet


def init_model(args):
    config_dict_file = open(f'{args.config_dir}/config.json', 'r')
    config_dict = json.load(config_dict_file)

    if args.model == 'ResNet' or args.model == 'SEResNet' or args.model == 'VGG':
        model_name = f'{args.model}{args.model_version}'
        return eval(f'{config_dict[model_name][args.dataset]}()')
    else:
        return eval(f'{config_dict[args.model][args.dataset]}()')

def init_dataset(args):
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
    elif args.dataset == 'HAR':
        train_data = HAR(args.data_dir, transform=torchvision.transforms.ToTensor(), train=True)
        test_data = HAR(args.data_dir, transform=torchvision.transforms.ToTensor(), train=False)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.train_bsz, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bsz, shuffle=False)

        return train_loader, test_loader