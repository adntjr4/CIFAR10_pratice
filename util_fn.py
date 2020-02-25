import time
import datetime

import torch

def print_mid_train_msg(epoch, batch, loss):
    print('[epoch : %d, batch: %5d] loss: %.3f \t\t\t\t\t' % (epoch, batch, loss))

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

def get_classification_test_result(device, net, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            predicted = net.get_response(device, images)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total
