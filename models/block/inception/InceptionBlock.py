import torch
from torch import nn
import torch.nn.functional as f

from models.BaseModel import BaseModel


class InceptionBlock(BaseModel):

    def __init__(self, in_channel, oc1, oc2, oc3, oc4):
        super(InceptionBlock, self).__init__()
        self.path1_1 = nn.Conv2d(in_channel, oc1, kernel_size=1)

        self.path2_1 = nn.Conv2d(in_channel, oc2[0], kernel_size=1)
        self.path2_2 = nn.Conv2d(oc2[0], oc2[1], kernel_size=3, padding=1)

        self.path3_1 = nn.Conv2d(in_channel, oc3[0], kernel_size=1)
        self.path3_2 = nn.Conv2d(oc3[0], oc3[1], kernel_size=5, padding=2)

        self.path4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.path4_2 = nn.Conv2d(in_channel, oc4, kernel_size=1)

    def forward(self, x):

        x_path1 = f.relu(self.path1_1(x))
        x_path2 = f.relu(self.path2_2(f.relu(self.path2_1(x))))
        x_path3 = f.relu(self.path3_2(f.relu(self.path3_1(x))))
        x_path4 = f.relu(self.path4_2(self.path4_1(x)))

        return torch.cat((x_path1, x_path2, x_path3, x_path4), dim=1)