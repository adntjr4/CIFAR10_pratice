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
MobileNet structure
> see "MobileNets: Efficient Convolutional Neural Networks for Mobile VisionApplications"
'''

class MobileNet(nn.Module):

    def __init__(self):
        super(MobileNet, self).__init__()

        self.init_conv = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # out_channels, stride
        block_layer = [ (64, 1),
                        (128, 2),
                        (128, 1),
                        (256, 2),
                        (256, 1),
                        (512, 2),
                        (512, 1),
                        (512, 1),
                        (512, 1),
                        (512, 1),
                        (512, 1),
                        (512, 1),
                        (1024, 2),
                        (1024, 1) ] # total 13 units -> 1 + 13x2 + 1 =  26 layer
        in_channels = 32
        layers = []
        for out_channels, stride in block_layer:
            layers.append(ConvUnit(in_channels, out_channels, stride))
            in_channels = out_channels
        self.layers = nn.Sequential(*layers)

        self.FNN = nn.Linear(1024, 10)

    def forward(self, x):
        out = self.init_conv(x)
        out = self.bn(out)
        out = self.relu(out)

        out = self.layers(out)

        out = F.adaptive_avg_pool2d(out, (1,1))
        out = out.view(out.size(0), -1)

        out = self.FNN(out)
        #out = F.softmax(out, dim=1)

        return out

class ConvUnit(nn.Module):

    def __init__(self, in_channels, out_channels, dw_stride=1): 
        '''
        dw_stride : depthwise conv layer stride (pointwise layer's stride is always 1)
        '''
        super(ConvUnit, self).__init__()
        self.dw_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=dw_stride, padding=1, groups=in_channels, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # Figure 3. in paper
        out = self.dw_conv(x)
        out = self.dw_bn(out)
        out = self.relu(out)
        out = self.pw_conv(out)
        out = self.pw_bn(out)
        out = self.relu(out)

        return out



