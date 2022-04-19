import torch.nn as nn

from models.BaseModel import BaseModel


class Bottleneck(BaseModel):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        # Downsample
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.relu(residual)

        residual = self.conv2(residual)
        residual = self.bn2(residual)
        residual = self.relu(residual)

        residual = self.conv3(residual)
        residual = self.bn3(residual)

        if self.downsample is not None:
            x = self.downsample(x)

        res = residual + x
        res = self.relu(res)

        return res
