from torch import  nn

from models.BaseModel import BaseModel
import torch.nn.functional as f


class AlexNet_CIFAR10(BaseModel):

    def __init__(self):
        super(AlexNet_CIFAR10, self).__init__()

        self.alex_net = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=1, padding=5), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), nn.Flatten(),
            nn.Linear(6144, 4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(4096, 10))

    def forward(self, x):
        # for layer in self.alex_net:
        #     x = layer(x)
        #     print(layer.__class__.__name__, f'Output shape: {x.shape}')

        x = self.alex_net(x)
        return f.log_softmax(x, dim=1)