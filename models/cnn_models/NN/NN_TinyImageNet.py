from torch import nn
import torch.nn.functional as f
from models.BaseModel import BaseModel

class NN_TinyImageNet(BaseModel):
    """ NN Network architecture. """
    def __init__(self):
        super(NN_TinyImageNet, self).__init__()
        self.fc1 = nn.Linear(64 * 64 * 3, 1024)
        self.fc2 = nn.Linear(1024, 200)

    def forward(self, x):
        x = x.view(-1, 64 * 64 * 3)
        x = f.relu(self.fc1(x))
        x = f.dropout(x, training=self.training)
        x = self.fc2(x)
        return f.log_softmax(x, dim=1)