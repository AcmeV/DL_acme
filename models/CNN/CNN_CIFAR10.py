from torch import nn
import torch.nn.functional as f
from models.BaseModel import BaseModel

class CNN_CIFAR10(BaseModel):
    """ CNN Network architecture. """
    def __init__(self):
        super(CNN_CIFAR10, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 100, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(100), nn.ReLU(),
            nn.Conv2d(100, 200, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(200), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Dropout(),
            nn.Linear(200, 10)
        )

    def forward(self, x):
        for layer in self.cnn:
            x = layer(x)
            print(layer.__class__.__name__, f'Output shape: {x.shape}')

        x = self.cnn(x)
        return f.log_softmax(x, dim=1)