from torch import nn
import torch.nn.functional as f
from models.BaseModel import BaseModel

class CNN_CIFAR100(BaseModel):
    """ CNN Network architecture. """
    def __init__(self):
        super(CNN_CIFAR100, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=2), nn.BatchNorm2d(96), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(96, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=4, padding=1), nn.Flatten(),
            nn.Linear(4096, 640), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(640, 100))

    def forward(self, x):
        x = self.cnn(x)
        return f.log_softmax(x, dim=1)