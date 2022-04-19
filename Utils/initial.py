import torch
import torchvision

import models
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
    elif args.dataset == 'CIFAR10':
        train_data = torchvision.datasets.CIFAR10(root=f'{args.data_dir}', train=True,
                                                download=True, transform=torchvision.transforms.ToTensor())
        test_data = torchvision.datasets.CIFAR10(root=f'{args.data_dir}', train=False,
                                               download=True, transform=torchvision.transforms.ToTensor())
    elif args.dataset == 'TinyImageNet':
        # 模拟 3 * 224 * 224
        # normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])
        # transform = torchvision.transforms.Compose([
        #     torchvision.transforms.RandomSizedCrop(224),
        #     torchvision.transforms.RandomHorizontalFlip(),
        #     torchvision.transforms.ToTensor(),
        #     normalize])

        # 自定义Dataset
        # transform = torchvision.transforms.ToTensor()
        #
        # train_data = TinyImageNet(f'{args.data_dir}/tiny-imagenet', transform=transform, train=True)
        # test_data = TinyImageNet(f'{args.data_dir}/tiny-imagenet', transform=transform, train=False)

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                             std=[0.2302, 0.2265, 0.2262])
        ])

        train_data = torchvision.datasets.ImageFolder(
            root=f'{args.data_dir}/TinyImageNet/train',transform=transform)
        test_data = torchvision.datasets.ImageFolder(
            root=f'{args.data_dir}/TinyImageNet/val',transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.train_bsz, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bsz, shuffle=False)

    return train_loader, test_loader