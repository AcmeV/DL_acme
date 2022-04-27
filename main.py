import argparse

from Utils.train.cnn_train import cnn_train
from Utils.train.rnn_train import rnn_train

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
parser.add_argument('--model-type', type=str, default='rnn',
                    choices=('cnn', 'rnn'))
parser.add_argument('--model', type=str, default='NN',
                    choices=('NN', 'CNN', 'AlexNet', 'NiN', 'VGG',
                             'GoogLeNet', 'ResNet', 'SEResNet',
                             'MyRNN', 'RNN'))
parser.add_argument('--model-version', type=int, default=18,
                    choices=(18, 34, 50, 101, 152,
                             11, 13, 16, 19),
                    help='version for ResNet or VGG,'
                         '(18, 34, 50, 101, 152) is for ResNet,'
                         '(11, 13, 16, 19) is for VGG')
# Dataset
parser.add_argument('--dataset', type=str, default='MNIST',
                    choices=('MNIST', 'CIFAR10', 'TinyImageNet',
                             'TimeMachine'))
parser.add_argument('--train-bsz', type=int, default=128)
parser.add_argument('--test-bsz', type=int, default=128)

# Hyper-parameters
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1)
parser.add_argument('--lr-decay', type=int, default=0,
                    choices=(0, 1))
parser.add_argument('--optim', type=str, default='SGD',
                    choices=('SGD', 'Adam'))
# RNN Settings
parser.add_argument('--num-step', type=int, default=35)
args = parser.parse_args()

if __name__ == '__main__':

    if args.model_type == 'cnn':
        cnn_train(args)
    elif args.model_type == 'rnn':
        rnn_train(args)

