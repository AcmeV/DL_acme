from torch import nn
import torch.nn.functional as f
from models import BaseModel
from models.block.vgg import VGGBlock_Same, VGGBlock, VGGBlock_Sub1


class VGG_CIFAR10(BaseModel):

    def __init__(self, conv_arch, in_channel=3 , size=32, num_classes=10):
        super(VGG_CIFAR10, self).__init__()

        out_channels = conv_arch[(len(conv_arch) - 1)][1]

        self.vgg_net = nn.Sequential(
            self._make_layer(in_channel, conv_arch), nn.Flatten(),
            nn.Linear(out_channels * 3 * 3, 4096),
            nn.ReLU(), nn.Dropout(0.5), nn.Linear(4096, 4096),
            nn.ReLU(), nn.Dropout(0.5), nn.Linear(4096, num_classes))

    def _make_layer(self, in_channel,  conv_arch):
        layers = []
        for idx, (num_convs, out_channel) in enumerate(conv_arch):
            if idx == 0 or idx == 1:
                layers.append(VGGBlock_Sub1(num_convs, in_channel, out_channel))
                in_channel = out_channel
            else:
                layers.append(VGGBlock(num_convs, in_channel, out_channel))
                in_channel = out_channel

        return nn.Sequential(*layers)

    def forward(self, x):
        # for debug
        # for layer in self.vgg_net:
        #     x = layer(x)
        #     print(layer.__class__.__name__, f'Output shape: {x.shape}')

        x = self.vgg_net(x)

        return f.log_softmax(x, dim=1)

def VGG11_CIFAR10():
    conv_arch = [(1, 64), (1, 128), (2, 256), (2, 512), (2, 512)]
    return VGG_CIFAR10(conv_arch, in_channel=3, size=32, num_classes=10)

def VGG13_CIFAR10():
    conv_arch = [(2, 64), (2, 128), (2, 256), (2, 512), (2, 512)]
    return VGG_CIFAR10(conv_arch, in_channel=3, size=32, num_classes=10)

def VGG16_CIFAR10():
    conv_arch = [(2, 64), (2, 128), (3, 256), (3, 512), (3, 512)]
    return VGG_CIFAR10(conv_arch, in_channel=3, size=32, num_classes=10)

def VGG19_CIFAR10():
    conv_arch = [(2, 64), (2, 128), (4, 256), (4, 512), (4, 512)]
    return VGG_CIFAR10(conv_arch, in_channel=3, size=32, num_classes=10)