import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models

import math

class Identity(nn.Module):
    """Identity Module"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class ConvNet(nn.Module):
    """LeNet++ as described in the Center Loss paper."""
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.resnet = models.resnet50()
        self.resnet.fc = Identity()
        self.final = nn.Linear(in_features=2048, out_features=3, bias=True)

    def forward(self, x):
        x = self.resnet(x)
        y = self.final(x)

        return x, y

__factory = {
    'cnn': ConvNet,
}

def create(name, num_classes):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](num_classes)

if __name__ == '__main__':
    pass
