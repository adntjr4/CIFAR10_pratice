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
CIFAR-10 데이터를 다루기 위한 pytorch 기본 예제
기본 CNN파일을 변형하여 DenseNet 구성
'''

class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()

    def custom_training(self, device, trainloader, criterion, optimizer, total_epoch=1, print_batch=2000, start_time=time.time(), path="./", name='test'): 
        progress_time = start_time
        for epoch in range(total_epoch):   # 데이터셋을 수차례 반복합니다.
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
                inputs, labels = data[0].to(device), data[1].to(device)

                # 변화도(Gradient) 매개변수를 0으로 만들고
                optimizer.zero_grad()

                # 순전파 + 역전파 + 최적화를 한 후
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # 통계를 출력합니다.
                running_loss += loss.item()
                if i % print_batch == print_batch-1:    # print every 2000 mini-batches
                    print_mid_train_msg(epoch, i+1, running_loss/print_batch)
                    running_loss = 0.0

                if i % 50 == 0:
                    if time.time()-progress_time > 1:
                        progress_time = time.time()
                        print_prog_msg(start_time, total_epoch, epoch, len(trainloader), i+1)
                

        print_finish_training_msg(start_time)
        torch.save(self.state_dict(), path+'CNN'+'_'+name)

    def get_response(self, device, data):
        data.to(device)
        outputs = self(data)
        _, predicted = torch.max(outputs.data, 1)
        return predicted

class CNNNet_2layer(CNNNet):
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
