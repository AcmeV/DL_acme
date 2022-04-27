from torch import  nn

from models.BaseModel import BaseModel
import torch.nn.functional as f

from models.cnn_models.block.nin import NiNBlock


class NiN_TinyImageNet(BaseModel):

    def __init__(self):
        super(NiN_TinyImageNet, self).__init__()

        self.nin_net = nn.Sequential(
            NiNBlock(3, 96, kernel_size=11, stride=1, padding=5),
            nn.MaxPool2d(kernel_size=3, stride=2),
            NiNBlock(96, 256, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            NiNBlock(256, 384, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            NiNBlock(384, 384, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2), nn.Dropout(0.5),
            NiNBlock(384, 200, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten())

    def forward(self, x):
        x = self.nin_net(x)
        return f.log_softmax(x, dim=1)