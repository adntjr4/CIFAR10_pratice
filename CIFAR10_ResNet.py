import time
import datetime

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

'''
https://github.com/KellerJordan/ResNet-PyTorch-CIFAR10/blob/master/model.py
을 참고하여 ResNet 구성
'''

class ResNet(nn.Module):
    def __init__(self, res_option='A', use_dropout=False):
        super(ResNet, self).__init__()
        self.res_option = res_option
        self.use_dropout = use_dropout

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1_bn = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)

        self.layers1 = self._make_layer(layer_count=5, in_channels=16, out_channels=16, stride=1)
        self.layers2 = self._make_layer(layer_count=5, in_channels=16, out_channels=32, stride=2)
        self.layers3 = self._make_layer(layer_count=5, in_channels=32, out_channels=64, stride=2)

        self.avgpool = nn.AvgPool2d(8, stride=1)

        self.linear = nn.Linear(64, 10)

    def _make_layer(self, layer_count, in_channels, out_channels, stride):
        return nn.Sequential(
            ResBlock(in_channels, out_channels, stride),
            *[ResBlock(out_channels, out_channels) for _ in range(layer_count-1)])

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv1_bn(out)
        out = self.relu1(out)

        out = self.layers1(out)
        out = self.layers2(out)
        out = self.layers3(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ResBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        
        # uses 1x1 convolutions for downsampling
        if in_channels != out_channels:
            self.projection = IdentityPadding(out_channels, in_channels, stride)
        else:
            self.projection = None

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.projection:
            residual = self.projection(x)
        out += residual
        out = self.relu(out)
        return out

class IdentityPadding(nn.Module):
    def __init__(self, num_filters, channels_in, stride):
        super(IdentityPadding, self).__init__()

        self.identity = nn.MaxPool2d(1, stride=stride)
        self.num_zeros = num_filters - channels_in
    
    def forward(self, x):
        out = F.pad(x, (0, 0, 0, 0, 0, self.num_zeros))
        out = self.identity(out)
        return out

if __name__ == '__main__':
    # here main function starts.
    pass
