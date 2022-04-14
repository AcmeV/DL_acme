import torch
import torchvision
from models.CNN_MNIST import CNN_MNIST
from models.NN_MNIST import ANN_MNIST


def init_model(args):
    if args.model == 'CNN' and args.dataset == 'mnist':
        return CNN_MNIST()
    elif args.model == 'ANN' and args.dataset == 'mnist':
        return ANN_MNIST()

def init_dataset(args):
    train_data, test_data = None, None
    if args.dataset == 'mnist':
        train_data = torchvision.datasets.MNIST(root='./data', train=True,
                                                download=True, transform=torchvision.transforms.ToTensor())
        test_data = torchvision.datasets.MNIST(root='./data', train=False,
                                               download=True, transform=torchvision.transforms.ToTensor())
    elif args.dataset == 'cifar10':
        train_data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=torchvision.transforms.ToTensor())
        test_data = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=torchvision.transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.train_bsz, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bsz, shuffle=False)

    return train_loader, test_loader