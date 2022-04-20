from torch import nn
import torch.nn.functional as f
from models import BaseModel
from models.block.vgg import VGGBlock


class VGG_TinyImageNet(BaseModel):

    def __init__(self, conv_arch, in_channel=3):
        super(VGG_TinyImageNet, self).__init__()

        self.vgg_net = nn.Sequential(
            self._make_layer(in_channel, conv_arch), nn.Flatten(),
            nn.Dropout(), nn.Linear(512 * 2 * 2, 512), nn.ReLU(True),
            nn.Dropout(), nn.Linear(512, 512), nn.ReLU(True),
            nn.Linear(512, 200))

    def _make_layer(self, in_channel,  conv_arch):
        layers = []
        for idx, (num_convs, out_channel) in enumerate(conv_arch):
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

def VGG11_TinyImageNet():
    conv_arch = [(1, 64), (1, 128), (2, 256), (2, 512), (2, 512)]
    return VGG_TinyImageNet(conv_arch, in_channel=3)

def VGG13_TinyImageNet():
    conv_arch = [(2, 64), (2, 128), (2, 256), (2, 512), (2, 512)]
    return VGG_TinyImageNet(conv_arch, in_channel=3)

def VGG16_TinyImageNet():
    conv_arch = [(2, 64), (2, 128), (3, 256), (3, 512), (3, 512)]
    return VGG_TinyImageNet(conv_arch, in_channel=3)

def VGG19_TinyImageNet():
    conv_arch = [(2, 64), (2, 128), (4, 256), (4, 512), (4, 512)]
    return VGG_TinyImageNet(conv_arch, in_channel=3)