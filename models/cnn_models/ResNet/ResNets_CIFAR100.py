import math

from torch import nn

from models.BaseModel import BaseModel
from models.cnn_models.block.residual.BasicBlock import BasicBlock
from models.cnn_models.block import Bottleneck
from models.cnn_models.block.residual.SE_BasicBlock import SE_BasicBlock
from models.cnn_models.block.residual.SE_Bottleneck import SE_Bottleneck

class ResNets_CIFAR100(BaseModel):

    def __init__(self, block, layers):
        num_classes = 100
        self.inplanes = 64

        super(ResNets_CIFAR100, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def ResNet18_CIFAR100():
    """Constructs a SE-ResNet-18 model.

        Args:
            num_classes = 10 (default)
        """
    model = ResNets_CIFAR100(BasicBlock, [2, 2, 2, 2])
    return model

def ResNet34_CIFAR100():
    """Constructs a SE-ResNet-34 model.

        Args:
            num_classes = 10 (default)
        """
    model = ResNets_CIFAR100(BasicBlock, [3, 4, 6, 3])
    return model

def ResNet50_CIFAR100():
    """Constructs a SE-ResNet-50 model.

    Args:
        num_classes = 10 (default)
    """
    model = ResNets_CIFAR100(Bottleneck, [3, 4, 6, 3])
    return model

def ResNet101_CIFAR100():
    """Constructs a SE-ResNet-101 model.

    Args:
        num_classes = 10 (default)
    """
    model = ResNets_CIFAR100(Bottleneck, [3, 4, 23, 3])
    return model

def ResNet152_CIFAR100():
    """Constructs a SE-ResNet-152 model.

    Args:
        num_classes = 10 (default)
    """
    model = ResNets_CIFAR100(Bottleneck, [3, 8, 36, 3])
    return model

def SEResNet18_CIFAR100():
    """Constructs a SE-ResNet-18 model.

        Args:
            num_classes = 10 (default)
        """
    model = ResNets_CIFAR100(SE_BasicBlock, [2, 2, 2, 2])
    return model

def SEResNet34_CIFAR100():
    """Constructs a SE-ResNet-34 model.

        Args:
            num_classes = 10 (default)
        """
    model = ResNets_CIFAR100(SE_BasicBlock, [3, 4, 6, 3])
    return model

def SEResNet50_CIFAR100():
    """Constructs a SE-ResNet-50 model.

    Args:
        num_classes = 10 (default)
    """
    model = ResNets_CIFAR100(SE_Bottleneck, [3, 4, 6, 3])
    return model

def SEResNet101_CIFAR100():
    """Constructs a SE-ResNet-101 model.

    Args:
        num_classes = 10 (default)
    """
    model = ResNets_CIFAR100(SE_Bottleneck, [3, 4, 23, 3])
    return model

def SEResNet152_CIFAR100():
    """Constructs a SE-ResNet-152 model.

    Args:
        num_classes = 10 (default)
    """
    model = ResNets_CIFAR100(SE_Bottleneck, [3, 8, 36, 3])
    return model