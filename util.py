import time
import datetime

import torch
import numpy as np

# mixup from https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py
def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]

    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# progress message print
def print_prog_msg(txt, start_time, total_epoch, epoch, total_batch, batch):
    elapsed = time.time() - start_time
    pg_per = (100/total_epoch) * (epoch + (batch+1)/(total_batch))
    total = 100*elapsed/pg_per
    remain = total - elapsed

    elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))
    remain_str = str(datetime.timedelta(seconds=int(remain)))
    total_str = str(datetime.timedelta(seconds=int(total)))
    print('>>> %s : %.2f%%, elapsed: %s, remaining: %s, total: %s \t\t\t\t\t' % (txt, pg_per, elapsed_str, remain_str, total_str), end='\r')

def print_finish_training_msg(start_time):
    total = time.time() - start_time
    total_str = str(datetime.timedelta(seconds=int(total)))
    print('Finished Training >>> (total elapsed time : %s) \t\t\t\t\t' % total_str)

from CIFAR10_CNN import CNNNet_2layer, CNNNet_3layer
from CIFAR10_ResNet import ResNet
from CIFAR10_DenseNet import DenseNet
from CIFAR10_PyramidalNet import PyramidalNet
from CIFAR10_MobileNet import MobileNet

def return_model_name(net):
    if isinstance(net, CNNNet_2layer) or isinstance(net, CNNNet_3layer):
        return 'CNN'
    elif isinstance(net, ResNet):
        return 'ResNet'
    elif isinstance(net, DenseNet):
        return 'DenseNet'
    elif isinstance(net, PyramidalNet):
        return 'PyramidalNet'
    elif isinstance(net, MobileNet):
        return 'MobileNet'
    else:
        raise Exception('Unknown network model instance')

