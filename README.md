# Pytorch practice using CIFAR-10 data set

## Reference papers, blogs and others

### __ResNet__  
[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)  

https://dnddnjs.github.io/cifar10/2018/10/09/resnet/  

### __DenseNet__  
[Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)

### __PyramidNet__  
[Deep Pyramidal Residual Networks](https://arxiv.org/abs/1610.02915)  

### __MobileNet__
[MobileNets: Efficient Convolutional Neural Networks for Mobile VisionApplications](https://arxiv.org/abs/1704.04861)  

### __MobileNetv2__
[MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)  

### __Mixup__  
[Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/abs/1812.01187)

https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py  

## Result

| Model                     | Error(%) | Params | Training time | Options                                  |
|---------------------------|:--------:|:------:|:-------------:|------------------------------------------|
| MobileNetV2               |   7.09   | 2.24 M |   03:58:39    | 300 epoch, w/ mixup                      |
| MobileNetV2               |   7.49   | 2.24 M |   03:46:45    | 300 epoch                                |
| MobileNet                 |   6.74   | 3.49 M |   02:43:13    | 300 epoch, w/ mixup                      |
| MobileNet                 |   8.07   | 3.49 M |   02:43:01    | 300 epoch                                |
| PyramidalNet (a=48,L=110) |   4.06   |        |               | 300 epoch, w/ mixup                      |
| PyramidalNet (a=48,L=110) |   5.27   |        |               | 300 epoch                                |
| DenseNet-BC (k=12,L=100)  |   4.65   |        |               | 300 epoch, w/ mixup                      |
| DenseNet-BC (k=12,L=100)  |   6.04   |        |               | 100 epoch, w/ PyramidalNet residual unit |
| DenseNet-BC (k=12,L=100)  |   5.57   |        |               | 300 epoch                                |
| DenseNet (k=12, L=40)     |   8.68   |        |               | 80 epoch                                 |
| ResNet-32                 |   8.20   |        |               | 80 epoch                                 |
| default-CNN               |   > 30   |        |               |                                          |

#### Note) All training was done by single RTX 2060 super with Ryzen 2600
