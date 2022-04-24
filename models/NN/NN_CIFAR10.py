from torch import nn
import torch.nn.functional as f
from models.BaseModel import BaseModel

class NN_CIFAR10(BaseModel):
    """ NN Network architecture. """
    def __init__(self):
        super(NN_CIFAR10, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = f.relu(self.fc1(x))
        x = f.dropout(x, training=self.training)
        x = self.fc2(x)
        return f.log_softmax(x, dim=1)