from torch import nn

from models.BaseModel import BaseModel


class VGGBlock(BaseModel):

    def __init__(self, num_conv, in_channel, out_channel):
        super(VGGBlock, self).__init__()
        layers = []

        for _ in range(num_conv):
            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channel = out_channel
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.vgg_block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.vgg_block(x)

        return x

class VGGBlock_Sub1(BaseModel):

    def __init__(self, num_conv, in_channel, out_channel):
        super(VGGBlock_Sub1, self).__init__()
        layers = []

        for _ in range(num_conv):
            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channel = out_channel
        layers.append(nn.MaxPool2d(kernel_size=2, stride=1))
        self.vgg_block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.vgg_block(x)

        return x

class VGGBlock_Sub2(BaseModel):

    def __init__(self, num_conv, in_channel, out_channel):
        super(VGGBlock_Sub2, self).__init__()
        layers = []

        for _ in range(num_conv):
            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channel = out_channel
        layers.append(nn.MaxPool2d(kernel_size=3, stride=1))
        self.vgg_block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.vgg_block(x)

        return x

class VGGBlock_Same(BaseModel):

    def __init__(self, num_conv, in_channel, out_channel):
        super(VGGBlock_Same, self).__init__()
        layers = []

        for _ in range(num_conv):
            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channel = out_channel
        layers.append(nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
        self.vgg_block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.vgg_block(x)

        return x