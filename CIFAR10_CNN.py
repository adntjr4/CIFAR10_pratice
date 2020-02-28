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
CIFAR-10 데이터를 다루기 위한 pytorch 기본 예제
기본 CNN을 사용함.
'''

class CNNNet_2layer(nn.Module):
    '''
    5x5 filter(20ch) -> max_pool -> 3x3 filter(128ch) -> max_pool -> (128->128->64->10) FNN
    '''
    def __init__(self):
        super(CNNNet_2layer, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5)
        self.conv1_bn = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, 128, 3)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # convolution 1
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # convolution 2
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # fully-connected network
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = F.log_softmax(self.fc3(x), dim=1)

        return output

class CNNNet_3layer(nn.Module):
    '''
    5x5 filter(20ch) -> max_pool -> 3x3 filter(128ch) -> max_pool -> 3x3 filter(512ch) -> max_pool -> (512->128->64->10) FNN
    '''
    def __init__(self):
        super(CNNNet_3layer, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5)
        self.conv2 = nn.Conv2d(20, 128, 3)
        self.conv3 = nn.Conv2d(128, 512, 3)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(512 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # 28x28
        x = F.max_pool2d(x, 2)      # 14x14
        x = F.relu(self.conv2(x))   # 12x12
        x = F.max_pool2d(x, 2)      # 6x6
        x = F.relu(self.conv3(x))   # 4x4
        x = F.max_pool2d(x, 2)      # 2x2
        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        output = F.log_softmax(self.fc3(x), dim=1)

        return output

class CNNNet_6layer(nn.Module):
    def __init__(self):
        super(CNNNet_6layer, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 3)
        self.conv2 = nn.Conv2d(20, 20, 3)
        self.conv3 = nn.Conv2d(20, 20, 3)
        self.conv4 = nn.Conv2d(20, 20, 3)
        self.conv5 = nn.Conv2d(20, 20, 3)
        self.conv6 = nn.Conv2d(20, 20, 3)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(20 * 10 * 10, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # 30x30
        x = F.relu(self.conv2(x))   # 28x28
        x = F.relu(self.conv3(x))   # 26x26
        x = F.relu(self.conv4(x))   # 24x24
        x = F.relu(self.conv5(x))   # 22x22
        x = F.relu(self.conv6(x))   # 20x20
        x = F.max_pool2d(x, 2)      # 10x10
        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        output = F.log_softmax(self.fc3(x), dim=1)

        return output

if __name__ == '__main__':
    # here main function starts.
    pass
