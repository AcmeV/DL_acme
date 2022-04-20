from torch import nn
import torch.nn.functional as f
from models.BaseModel import BaseModel

class CNN_TinyImageNet(BaseModel):
    """ CNN Network architecture. """
    def __init__(self):
        super(CNN_TinyImageNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 200, kernel_size=3, stride=4), nn.BatchNorm2d(200), nn.ReLU(),
            nn.Conv2d(200, 400, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(400), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Dropout(),
            nn.Linear(400, 200)
        )

    def forward(self, x):
        # for layer in self.cnn:
        #     x = layer(x)
        #     print(layer.__class__.__name__, f'Output shape: {x.shape}')

        x = self.cnn(x)
        return f.log_softmax(x, dim=1)