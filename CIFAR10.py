import time
import datetime
import os

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

import CIFAR10_CNN as CNN
from util_fn import get_classification_test_result

'''
CIFAR-10 데이터 셋을 다루고 다른 Net을 호출하는 python file
'''

# default setting option
model_dir = './model/'
model_type_list = ('CNN', 'ResNet')
model_type = 0
model_name = '3layer(2epoch)'
execute = 'train'

## training option
total_epoch = 2
train_sample_num = 50000
test_sample_num = 10000
batch_size = 4

core_idx = 0
DEVICE_LIST = ("cpu", "cuda:0")


# device
device = torch.device(DEVICE_LIST[core_idx] if torch.cuda.is_available() else "cpu")

# Net

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = CNN.CNNNet_3layer()
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

model_path = model_dir + model_type_list[model_type] + '_' + model_name
if os.path.isfile(model_path):
    print('Network was already trained.')
    net.load_state_dict(torch.load(model_path))
    net.eval()
else:
    net.custom_training(device=device, trainloader=trainloader, criterion=criterion, optimizer=optimizer, total_epoch=total_epoch, 
                        print_batch=2000, start_time=time.time(), path=model_dir, name=model_name)

Accuracy = get_classification_test_result(device, net, testloader)
print('Accuracy: %d %%' % Accuracy)

