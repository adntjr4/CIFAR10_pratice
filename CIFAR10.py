import time
import datetime
import os
import sys

import torch
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

from tensorboardX import SummaryWriter

from CIFAR10_CNN import CNNNet_2layer, CNNNet_3layer
from CIFAR10_ResNet import ResNet
from CIFAR10_DenseNet import DenseNet
from CIFAR10_PyramidalNet import PyramidalNet
from CIFAR10_MobileNet import MobileNet
from util import *

'''
CIFAR-10 데이터 셋을 다루고 다른 Net을 호출하는 python file
'''

# model option
model_dir = './model/'
model_name = '13block_mixup'

#net = DenseNet(growth_rate=12, layer_num=100, B_mode=True, C_mode=True, theta=0.5, P_block=False)
net = MobileNet()
net_name = return_model_name(net)

# training option
training_mode = True
load_epoch = 0 # (#epoch)

total_epoch = 300
batch_size = 128
initial_learning_rate = 0.1
momentum = 0.9
weight_decay = 1e-4

model_save_epoch = 10
mixup = False

# device
device_idx = 1
DEVICE_LIST = ("cpu", "cuda:0")
device = torch.device(DEVICE_LIST[device_idx] if torch.cuda.is_available() else "cpu")

# net
net.cuda()
net = torch.nn.DataParallel(net)
cudnn.benchmark = True
num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('Number of parameters : %d' % num_params)

# training

#transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transforms_train = transforms.Compose([
  transforms.RandomCrop(32, padding=4),
  transforms.RandomHorizontalFlip(),
  transforms.ToTensor(),
  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transforms_test = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transforms_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transforms_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=initial_learning_rate, momentum=momentum, weight_decay=weight_decay)
step_lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(total_epoch*0.5), int(total_epoch*0.75)], gamma=0.1)

model_path = model_dir + net_name + '_' + model_name

# logs
logfile = open('./logs/print_log/%s_%s_%s.log'%(datetime.datetime.now().strftime('%m.%d-%H.%M.%S'), net_name, model_name), 'w')
writer = SummaryWriter('./logs/%s'%(net_name+'_'+model_name))

def train(dataloader):
    net.train()
    start_time = time.time()
    progress_time = start_time

    for epoch in range(total_epoch):
        running_loss = 0.0

        correct = 0
        total = 0

        batch_num = len(dataloader)

        for i, data in enumerate(dataloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            if mixup:
                inputs, labels_a, labels_b, lam = mixup_data(inputs, labels)
                inputs, labels_a, labels_b = map(Variable, (inputs, labels_a, labels_b))

            outputs = net(inputs)

            if mixup:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)

            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ')
                exit(1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            if mixup:
                correct += (lam * predicted.eq(labels_a.data).cpu().sum().float()+ (1 - lam) * predicted.eq(labels_b.data).cpu().sum().float())
            else:
                correct += (predicted == labels).sum().item()

            if time.time()-progress_time > 1:
                progress_time = time.time()
                print_prog_msg('training', start_time, total_epoch, epoch, len(dataloader), i+1)

        step_lr_scheduler.step()
        train_accuracy = 100.0 * correct/total

        txt = "[epoch : %d] train_error : %.2f %%, loss : %.4f \t\t\t\t\t" % (epoch+1, 100.0-train_accuracy, running_loss/batch_num)
        print(txt)
        logfile.write(txt)

        if (epoch+1) % model_save_epoch == 0:
            torch.save(net.state_dict(), model_dir + net_name + '_' + model_name + '(%depoch)'%(epoch+1))
            test_accuracy = test(testloader)
            writer.add_scalar('log/test_error', 100.0-test_accuracy, epoch)
            txt = "[epoch : %d] test_error : %.2f %% \t\t\t\t\t" % (epoch+1, 100.0-test_accuracy)
            print(txt)
            logfile.write(txt)
        writer.add_scalar('log/train_error', 100.0-train_accuracy, epoch)

    print_finish_training_msg(start_time)
    torch.save(net.state_dict(), model_dir + net_name + '_' + model_name)

def test(dataloader):
    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        test_num = len(dataloader)
        for idx, data in enumerate(dataloader):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print('>>> testing : %3d/%3d' % (idx, test_num), end='\r')
    net.train()
    return 100.0 * correct / total

if __name__=='__main__':
    if training_mode:
        if os.path.isfile(model_path):
            #print('Network was already trained. continue training.')
            #net.load_state_dict(torch.load(model_path))
            print('There is already trained model')
            exit()
        train(trainloader)
    else:
        if load_epoch != 0:
            model_path += '(%depoch)'%load_epoch
        if os.path.isfile(model_path):
            print('Network was already trained.')
            net.load_state_dict(torch.load(model_path))
            net.eval()
        else:
            print('There is no trained model')
            exit()
            
    Accuracy = test(testloader)
    txt = 'Test error: %.2f %%\t\t\t' % (100.0-Accuracy)
    print(txt)
    logfile.write(txt)

logfile.close()
