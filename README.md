# Pytorch practice using CIFAR-10 data set

## Reference papers, blogs and others

### __ResNet__  
Deep Residual Learning for Image Recognition  
https://dnddnjs.github.io/cifar10/2018/10/09/resnet/  

### __DenseNet__  
Densely Connected Convolutional Networks  

### __PyramidNet__  
Deep Pyramidal Residual Networks

### __Mixup__  
Bag of Tricks for Image Classification with Convolutional Neural Networks  
https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py  

## Result

| Model                    | Error(%) |                                          |
|--------------------------|:--------:|------------------------------------------|
| PyramidalNet(a=48,L=110) |   5.27   | 300 epoch, w/ (d) residual unit          |
| DenseNet-BC(k=12,L=100)  |   4.65   | 300 epoch, w/ mixup                      |
| DenseNet-BC(k=12,L=100)  |   6.04   | 100 epoch, w/ PyramidalNet residual unit |
| DenseNet-BC(k=12,L=100)  |   5.57   | 300 epoch                                |
| DenseNet(k=12, L=40)     |   8.68   | 80 epoch                                 |
| ResNet-32                |   8.20   | 80 epoch                                 |
| default-CNN              |   > 30   |                                          |
