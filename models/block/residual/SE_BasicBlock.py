from torch import nn

from models.BaseModel import BaseModel
import torch.nn.functional as f


class SE_BasicBlock(BaseModel):

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(SE_BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        # SE
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_down = nn.Conv2d(planes * self.expansion, planes // self.expansion, kernel_size=1, bias=False)
        self.conv_up = nn.Conv2d(planes // self.expansion, planes * self.expansion, kernel_size=1, bias=False)
        self.sig = nn.Sigmoid()

        # Downsample
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = f.relu(self.bn1(self.conv1(x)))
        residual = self.bn2(self.conv2(residual))

        se = self.global_pool(residual)
        se = self.conv_down(se)
        se = self.relu(se)
        se = self.conv_up(se)
        se = self.sig(se)

        if self.downsample is not None:
            x = self.downsample(x)

        res = se * residual + x
        res = self.relu(res)

        return res