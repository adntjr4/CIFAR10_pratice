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
DenseNet code 작성
'''

class DenseNet(nn.Module):
    def __init__(self, growth_rate=12, layer_num=40, B_mode=False, C_mode=False, theta=0.5, P_block=False):
        '''
        growth_rate : k (in paper)
        layer_num : layer number of entire network
        B_mode : Bottle neck option (in paper)
        C_mode : Compression option (in paper)
        P_block : building block in PyramidalNet
        '''
        super(DenseNet, self).__init__()
        self.k = growth_rate

        # dafault setting : 3 dense block
        if (B_mode and (layer_num-4)%6 != 0) or ((not B_mode) and (layer_num-4)%3 != 0):
            raise Exception('layer 개수가 안 나눠떨어짐.')

        block_layer_num = (layer_num-4) // 6

        self.init_conv = nn.Conv2d(3, 2*self.k, kernel_size=3, stride=1, padding=0)

        block_in1 = 2*self.k
        self.block1 = DenseBlock(block_in1, block_layer_num, self.k, B_mode=B_mode, P_block=P_block)
        block_out2 = block_in1 + block_layer_num * self.k

        self.transition1 = TransitionLayer(block_out2, C_mode=C_mode, theta=theta)

        block_in = int(theta*block_out2) if C_mode else block_out2
        self.block2 = DenseBlock(block_in, block_layer_num, self.k, B_mode=B_mode, P_block=P_block)
        block_out = block_in + block_layer_num * self.k

        self.transition2 = TransitionLayer(block_out, C_mode=C_mode, theta=theta)

        block_in = int(theta*block_out) if C_mode else block_out
        self.block3 = DenseBlock(block_in, block_layer_num, self.k, B_mode=B_mode, P_block=P_block)
        block_out = block_in + block_layer_num * self.k

        self.FNN = nn.Linear(block_out, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        conv_out = self.init_conv(x)

        block1_out = self.block1(conv_out)
        transit1_out = self.transition1(block1_out)
        block2_out = self.block2(transit1_out)
        transit2_out = self.transition2(block2_out)
        block3_out = self.block3(transit2_out)

        avg_pooling_out = F.adaptive_avg_pool2d(block3_out, (1,1))
        avg_pooling_out = avg_pooling_out.view(avg_pooling_out.size(0), -1)

        out = self.FNN(avg_pooling_out)

        return out

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, C_mode=False, theta=0.5):
        super(TransitionLayer, self).__init__()

        out_channels = int(theta*in_channels) if C_mode else in_channels

        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.avg_pool(out)
        return out

class DenseBlock(nn.Sequential):
    def __init__(self, in_channels, layer_num, k, B_mode, P_block):
        super(DenseBlock, self).__init__()
        self.layer_num = layer_num if B_mode else 2*layer_num

        for i in range(layer_num):
            in_t = in_channels + k*i
            self.add_module('bottle_neck_%d'%i, BottleNeck(in_t, k, B_mode, P_block))

class BottleNeck(nn.Module):
    def __init__(self, in_channels, k, B_mode, P_block):
        super(BottleNeck, self).__init__()
        self.B_mode = B_mode
        self.P_block = P_block

        if not B_mode:
            if not self.P_block:
                self.bn = nn.BatchNorm2d(in_channels)
                self.relu = nn.ReLU(inplace=True)
                self.conv = nn.Conv2d(in_channels, k, kernel_size=3, stride=1, padding=1)
            else:
                self.bn1 = nn.BatchNorm2d(in_channels)
                self.relu = nn.ReLU(inplace=True)
                self.conv = nn.Conv2d(in_channels, k, kernel_size=3, stride=1, padding=1)
                self.bn2 = nn.BatchNorm2d(k)
        else:
            if not self.P_block:
                self.bn1 = nn.BatchNorm2d(in_channels)
                self.relu = nn.ReLU(inplace=True)
                self.conv1 = nn.Conv2d(in_channels, 4*k, kernel_size=1, stride=1, padding=0)
                self.bn2 = nn.BatchNorm2d(4*k)
                self.conv2 = nn.Conv2d(4*k, k, kernel_size=3, stride=1, padding=1)
            else:
                self.bn1 = nn.BatchNorm2d(in_channels)
                self.relu = nn.ReLU(inplace=True)
                self.conv1 = nn.Conv2d(in_channels, 4*k, kernel_size=1, stride=1, padding=0)
                self.bn2 = nn.BatchNorm2d(4*k)
                self.conv2 = nn.Conv2d(4*k, k, kernel_size=3, stride=1, padding=1)
                self.bn3 = nn.BatchNorm2d(k)
    
    def forward(self, x):
        if not self.B_mode:
            if not self.P_block:
                out = self.bn(x)
                out = self.relu(out)
                out = self.conv(out)
                out = torch.cat((x, out), dim=1)
            else:
                out = self.bn1(x)
                out = self.relu(out)
                out = self.conv(out)
                out = self.bn2(x)
                out = torch.cat((x, out), dim=1)
        else:
            if not self.P_block:
                out = self.bn1(x)
                out = self.relu(out)
                out = self.conv1(out)
                out = self.bn2(out)
                out = self.relu(out)
                out = self.conv2(out)
                out = torch.cat((x, out), dim=1)
            else:
                out = self.bn1(x)
                out = self.conv1(out)
                out = self.bn2(out)
                out = self.relu(out)
                out = self.conv2(out)
                out = self.bn3(out)
                out = torch.cat((x, out), dim=1)

        return out

if __name__ == '__main__':
    # here main function starts.
    pass
