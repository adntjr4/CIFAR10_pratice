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
MobileNetV2 structure
> see "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
'''

class MobileNetV2(nn.Module):

    def __init__(self):
        super(MobileNetV2, self).__init__()

        self.init_conv = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.init_bn = nn.BatchNorm2d(32)
        self.relu6 = nn.ReLU6(inplace=True)

        sequence_list = [   (1, 16,  1, 1),
                            (6, 24,  2, 2),
                            (6, 32,  3, 2),
                            (6, 64,  4, 2),
                            (6, 96,  3, 1),
                            (6, 160, 3, 2),
                            (6, 320, 1, 1) ]

        layers = []
        in_channels = 32
        for t, c, n, s in sequence_list:
            layers += self.make_sequence(in_channels, t, c, n, s)
            in_channels = c
        self.sequences = nn.Sequential(*layers)

        self.conv1 = nn.Conv2d(320, 1280, kernel_size=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(1280)
        self.FNN = nn.Linear(1280, 10)
        #self.conv2 = nn.Conv2d(1280, 10, kernel_size=1, padding=0, bias=False)
        #self.bn2 = nn.BatchNorm2d(10)

    def forward(self, x):
        out = self.init_conv(x)
        out = self.init_bn(out)
        out = self.relu6(out)

        out = self.sequences(out)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu6(out)

        out = F.adaptive_avg_pool2d(out, (1,1))

        #out = self.conv2(out)
        #out = self.bn2(out)
        #out = self.relu6(out)

        out = out.view(out.size(0), -1)

        out = self.FNN(out)

        return out

    def make_sequence(self, in_ch, t, c, n, s):
        layers = []
        layers.append(ConV2Unit(in_ch, c, s, t))
        for i in range(n-1):
            layers.append(ConV2Unit(c, c, 1, t))
        return layers

class ConV2Unit(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, expansion_factor=6): 
        super(ConV2Unit, self).__init__()

        self.t = expansion_factor
        self.s = stride

        if in_channels != out_channels:
            self.projection = IdentityPadding(out_channels, in_channels, stride)
        else:
            self.projection = None

        self.relu6 = nn.ReLU6(inplace=True)

        self.conv1 = nn.Conv2d(in_channels, in_channels*self.t, kernel_size=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels*self.t)
        self.dw_conv = nn.Conv2d(in_channels*self.t, in_channels*self.t, kernel_size=3, stride=self.s, padding=1, groups=in_channels*self.t, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_channels*self.t)
        self.conv2 = nn.Conv2d(in_channels*self.t, out_channels, kernel_size=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu6(out)

        out = self.dw_conv(out)
        out = self.dw_bn(out)
        out = self.relu6(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.s == 1:
            if self.projection:
                residual = self.projection(residual)
            out += residual

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
