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

from util_fn import *

'''
기본 CNN파일을 변형하여 DenseNet 구성
'''

class CNNNet_2layer(CustomNet):
    '''
    5x5 filter(20ch) -> max_pool -> 3x3 filter(128ch) -> max_pool -> (128->128->64->10) FNN
    '''
    def __init__(self):
        super(CNNNet_2layer, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5)
        self.conv2 = nn.Conv2d(20, 128, 3)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(128 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
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
