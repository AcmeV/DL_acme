from torch import nn
import torch.nn.functional as f
from models.BaseModel import BaseModel

class CNN_MNIST(BaseModel):
    """ CNN Network architecture. """
    def __init__(self):
        super(CNN_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=5)
        self.conv2 = nn.Conv2d(2, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # conv -> relu -> pooling -> conv -> relu -> pooling
        a_layer1 = f.relu(self.conv1(x))
        a_after_pooling1 = f.max_pool2d(a_layer1, 2)
        a_layer2 = f.relu(self.conv2_drop(self.conv2(a_after_pooling1)))
        a_after_pooling2 = f.max_pool2d(a_layer2, 2)

        # -> fc ->relu -> fc -> softmax
        x = a_after_pooling2.view(-1, 320)
        x = f.relu(self.fc1(x))
        x = f.dropout(x, training=self.training)
        x = self.fc2(x)
        return f.log_softmax(x, dim=1)