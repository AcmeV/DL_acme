import math

from torch import nn

from models.BaseModel import BaseModel
from models.cnn_models.block.residual.BasicBlock import BasicBlock
from models.cnn_models.block import Bottleneck
from models.cnn_models.block.residual.SE_BasicBlock import SE_BasicBlock
from models.cnn_models.block.residual.SE_Bottleneck import SE_Bottleneck

class ResNets_MNIST(BaseModel):

    def __init__(self, block, layers):
        num_classes = 10
        self.inplanes = 64

        super(ResNets_MNIST, self).__init__()
        self.resnet = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            self._make_layer(block, 64, layers[0]),
            self._make_layer(block, 128, layers[1], stride=2),
            self._make_layer(block, 256, layers[2], stride=2),
            self._make_layer(block, 512, layers[3], stride=2),
            nn.AvgPool2d(4), nn.Flatten(), nn.Dropout(),
            nn.Linear(512 * block.expansion, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.resnet(x)

        return x

def ResNet18_MNIST():
    """Constructs a SE-ResNet-18 model.

        Args:
            num_classes = 10 (default)
        """
    model = ResNets_MNIST(BasicBlock, [2, 2, 2, 2])
    return model

def ResNet34_MNIST():
    """Constructs a SE-ResNet-34 model.

        Args:
            num_classes = 10 (default)
        """
    model = ResNets_MNIST(BasicBlock, [3, 4, 6, 3])
    return model

def ResNet50_MNIST():
    """Constructs a SE-ResNet-50 model.

    Args:
        num_classes = 10 (default)
    """
    model = ResNets_MNIST(Bottleneck, [3, 4, 6, 3])
    return model

def ResNet101_MNIST():
    """Constructs a SE-ResNet-101 model.

    Args:
        num_classes = 10 (default)
    """
    model = ResNets_MNIST(Bottleneck, [3, 4, 23, 3])
    return model

def ResNet152_MNIST():
    """Constructs a SE-ResNet-152 model.

    Args:
        num_classes = 10 (default)
    """
    model = ResNets_MNIST(Bottleneck, [3, 8, 36, 3])
    return model

def SEResNet18_MNIST():
    """Constructs a SE-ResNet-18 model.

        Args:
            num_classes = 10 (default)
        """
    model = ResNets_MNIST(SE_BasicBlock, [2, 2, 2, 2])
    return model

def SEResNet34_MNIST():
    """Constructs a SE-ResNet-34 model.

        Args:
            num_classes = 10 (default)
        """
    model = ResNets_MNIST(SE_BasicBlock, [3, 4, 6, 3])
    return model

def SEResNet50_MNIST():
    """Constructs a SE-ResNet-50 model.

    Args:
        num_classes = 10 (default)
    """
    model = ResNets_MNIST(SE_Bottleneck, [3, 4, 6, 3])
    return model

def SEResNet101_MNIST():
    """Constructs a SE-ResNet-101 model.

    Args:
        num_classes = 10 (default)
    """
    model = ResNets_MNIST(SE_Bottleneck, [3, 4, 23, 3])
    return model

def SEResNet152_MNIST():
    """Constructs a SE-ResNet-152 model.

    Args:
        num_classes = 10 (default)
    """
    model = ResNets_MNIST(SE_Bottleneck, [3, 8, 36, 3])
    return model