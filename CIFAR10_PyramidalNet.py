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
PyramidalNet code 작성
'''

class PyramidalNet(nn.Module):

    def __init__(self, shape='add', alpha=48, layer_num=110, bottle_neck=False, block_mode='d'):
        '''
        shape : 'add' or 'mul' (add: additive PyramidalNet, mul: multiplicative PyramidalNet)
        alpha : widening step factor (α)
        layer_num : depth of network
        bottle_neck : bottle neck use of residual unit
        block_mode : newly provided residual unit mode ('a', 'b', 'c', 'd' in Figure 6.)
        '''
        super(PyramidalNet, self).__init__()

        if not shape in ('add', 'mul'):
            raise Exception('choose one mode between add(additive) and mul(multiplicative)')

        if not block_mode in ('a', 'b', 'c', 'd'):
            raise Exception('Wrong block mode.')
        
        if bottle_neck:
            if (layer_num-2)%9 != 0:
                raise Exception('layer_num is not dividable')
            block_unit_num = int((layer_num-2)/9)
        else:
            if (layer_num-2)%6 != 0:
                raise Exception('layer_num is not dividable')
            block_unit_num = int((layer_num-2)/6)

        self.init_conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)

        if shape == 'add':
            final_feature_channels = int(16 + alpha)
        elif shape == 'mul':
            final_feature_channels = int(16 * alpha)

        self.block1 = BuildingBlock(block_idx=0, block_unit_num=block_unit_num, alpha=alpha, shape=shape, bottle_neck=bottle_neck, block_mode=block_mode)
        self.block2 = BuildingBlock(block_idx=1, block_unit_num=block_unit_num, alpha=alpha, shape=shape, bottle_neck=bottle_neck, block_mode=block_mode)
        self.block3 = BuildingBlock(block_idx=2, block_unit_num=block_unit_num, alpha=alpha, shape=shape, bottle_neck=bottle_neck, block_mode=block_mode)

        self.FNN = nn.Linear(final_feature_channels, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        conv_out = self.init_conv(x)

        block1_out = self.block1(conv_out)
        block2_out = self.block2(block1_out)
        block3_out = self.block3(block2_out)

        avg_pooling_out = F.adaptive_avg_pool2d(block3_out, (1,1))
        avg_pooling_out = avg_pooling_out.view(avg_pooling_out.size(0), -1)

        out = self.FNN(avg_pooling_out)

        return out

class BuildingBlock(nn.Sequential):

    def __init__(self, block_idx, block_unit_num, alpha, shape, bottle_neck, block_mode):
        super(BuildingBlock, self).__init__()

        N = 3 * block_unit_num  # total unit number in network

        for unit_idx in range(block_unit_num):
            k = block_idx*block_unit_num + unit_idx

            if shape == 'add':
                in_channels = int(16 + alpha*k/N)
                out_channels = int(16 + alpha*(k+1)/N)
            elif shape == 'mul':
                in_channels = int(16 * (alpha**(k/N)))
                out_channels = int(16 * (alpha**((k+1)/N)))

            if unit_idx == 0 and block_idx != 0:
                self.add_module('residual_unit_%d'%k, Unit(in_channels, out_channels, stride=2, bottle_neck=bottle_neck, block_mode=block_mode))
            else:
                self.add_module('residual_unit_%d'%k, Unit(in_channels, out_channels, stride=1, bottle_neck=bottle_neck, block_mode=block_mode))


class Unit(nn.Module):

    def __init__(self, in_channels, out_channels, stride, bottle_neck, block_mode):
        super(Unit, self).__init__()
        self.bottle_neck = bottle_neck
        self.block_mode = block_mode
        self.padding_size = out_channels - in_channels

        self.relu = nn.ReLU(inplace=True)

        if not bottle_neck:
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            if block_mode in ('c', 'd'):
                self.bn3 = nn.BatchNorm2d(out_channels)
        else:
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, int(out_channels/4), kernel_size=3, stride=stride, padding=1)
            self.bn3 = nn.BatchNorm2d(out_channels)
            self.conv3 = nn.Conv2d(int(out_channels/4), out_channels, kernel_size=1, stride=1, padding=0)
            if block_mode in ('c', 'd'):
                self.bn4 = nn.BatchNorm2d(out_channels)

        if stride != 1:
            self.downsample = nn.AvgPool2d((2,2), stride = (2, 2))  
        else:
            self.downsample = None

    def forward(self, x):
        if not self.bottle_neck:
            out = self.bn1(x)
            if self.block_mode in ('a', 'c'):
                out = self.relu(out)
            out = self.conv1(out)
            out = self.bn2(out)
            out = self.relu(out)
            out = self.conv2(out)
            if self.block_mode in ('c', 'd'):
                out = self.bn3(out)
        else:
            out = self.bn1(x)
            if self.block_mode in ('a', 'c'):
                out = self.relu(out)
            out = self.conv1(out)
            out = self.bn2(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn3(out)
            out = self.relu(out)
            out = self.conv3(out)
            if self.block_mode in ('c', 'd'):
                out = self.bn4(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x
        residual = F.pad(residual, (0, 0, 0, 0, 0, self.padding_size))
        
        out += residual
        return out


if __name__ == '__main__':
    # here main function starts.
    pass
