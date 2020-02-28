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

from tensorboardX import SummaryWriter

from CIFAR10_CNN import CNNNet_2layer, CNNNet_3layer
from CIFAR10_ResNet import ResNet

'''
CIFAR-10 데이터 셋을 다루고 다른 Net을 호출하는 python file
'''

# default setting option
model_dir = './model/'
net_name = 'ResNet'
model_name = 'bch_sch_aug'
execute_train = True

## training option
total_epoch = 80
decay_epoch = [40, 60]

save_epoch = 10
train_sample_num = 50000
test_sample_num = 10000
batch_size = 128

# device
device_idx = 1
DEVICE_LIST = ("cpu", "cuda:0")
device = torch.device(DEVICE_LIST[device_idx] if torch.cuda.is_available() else "cpu")

# net
net = ResNet()
net.to(device)
num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

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

writer = SummaryWriter('./logs')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
step_lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay_epoch, gamma=0.1)

model_path = model_dir + net_name + '_' + model_name

def train(dataloader):
    net.train()
    start_time = time.time()
    progress_time = start_time
    batch_num = len(dataloader)

    for epoch in range(total_epoch):
        running_loss = 0.0

        correct = 0
        total = 0

        for i, data in enumerate(dataloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if time.time()-progress_time > 1:
                progress_time = time.time()
                print_prog_msg(start_time, total_epoch, epoch, len(dataloader), i+1)

        step_lr_scheduler.step()
        train_accuracy = 100.0 * correct/total

        print("[epoch : %d] train_accuracy : %.1f%%, loss : %.2f \t\t\t\t\t" % (epoch+1, train_accuracy, running_loss))
        if (epoch+1) % save_epoch == 0:
            torch.save(net.state_dict(), model_dir + net_name + '_' + model_name + '(%depoch)'%(epoch+1))
            test_accuracy = test(testloader)
            writer.add_scalar('log/test_error', test_accuracy, epoch)
            print("[epoch : %d] test_accuracy : %.1f%% \t\t\t\t\t" % (epoch+1, test_accuracy))
        writer.add_scalar('log/train_error', train_accuracy, epoch)

    print_finish_training_msg(start_time)
    #torch.save(self.state_dict(), path + self.model_name + '_' + name)

def test(dataloader):
    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    net.train()
    return 100.0 * correct / total

def print_prog_msg(start_time, total_epoch, epoch, total_batch, batch):
    elapsed = time.time() - start_time
    pg_per = (100/total_epoch) * (epoch + (batch+1)/(total_batch))
    total = 100*elapsed/pg_per
    remain = total - elapsed

    elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))
    remain_str = str(datetime.timedelta(seconds=int(remain)))
    total_str = str(datetime.timedelta(seconds=int(total)))
    print('>>> %.2f%%, elapsed: %s, remaining: %s, total: %s \t' % (pg_per, elapsed_str, remain_str, total_str), end='\r')

def print_finish_training_msg(start_time):
    total = time.time() - start_time
    total_str = str(datetime.timedelta(seconds=int(total)))
    print('Finished Training >>> (total elapsed time : %s) \t\t\t\t\t' % total_str)

if __name__=='__main__':
    if execute_train:
        if os.path.isfile(model_path):
            print('Network was already trained. continue training.')
            net.load_state_dict(torch.load(model_path))
        train(trainloader)
    else:
        if os.path.isfile(model_path):
            print('Network was already trained.')
            net.load_state_dict(torch.load(model_path))
            net.eval()
        else:
            print('There is no trained model')
            exit()
            
    Accuracy = test(testloader)
    print('Accuracy: %d %%' % Accuracy)
