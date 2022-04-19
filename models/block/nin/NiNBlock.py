from torch import nn

from models.BaseModel import BaseModel


class NiNBlock(BaseModel):

    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(NiNBlock, self).__init__()
        self.nin_block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size = kernel_size,
                      stride=stride, padding=padding),
            nn.ReLU(), nn.Conv2d(out_channel, out_channel, kernel_size=1),
            nn.ReLU(), nn.Conv2d(out_channel, out_channel, kernel_size=1),
            nn.ReLU())

    def forward(self, x):

        x = self.nin_block(x)

        return x