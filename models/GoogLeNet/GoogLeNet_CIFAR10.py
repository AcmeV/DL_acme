from torch import  nn

from models.BaseModel import BaseModel
import torch.nn.functional as f

from models.block.inception.InceptionBlock import InceptionBlock


class GoogLeNet_CIFAR10(BaseModel):

    def __init__(self):
        super(GoogLeNet_CIFAR10, self).__init__()

        stage1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        stage2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(),
            nn.Conv2d(64, 192, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        stage3 = nn.Sequential(
            InceptionBlock(192, 64, (96, 128), (16, 32), 32),
            InceptionBlock(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        stage4 = nn.Sequential(
            InceptionBlock(480, 192, (96, 208), (16, 48), 64),
            InceptionBlock(512, 160, (112, 224), (24, 64), 64),
            InceptionBlock(512, 128, (128, 256), (24, 64), 64),
            InceptionBlock(512, 112, (144, 288), (32, 64), 64),
            InceptionBlock(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        stage5 = nn.Sequential(
            InceptionBlock(832, 256, (160, 320), (32, 128), 128),
            InceptionBlock(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1,1)), nn.Flatten())

        self.goog_lenet = nn.Sequential(stage1, stage2, stage3, stage4,
                                 stage5 , nn.Linear(1024, 10))

    def forward(self, x):
        # for debug
        # for layer in self.goog_lenet:
        #     x = layer(x)
        #     print(layer.__class__.__name__, f'Output shape: {x.shape}')

        x = self.goog_lenet(x)
        return f.log_softmax(x, dim=1)