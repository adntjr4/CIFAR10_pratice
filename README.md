# Pytorch practice through CIFAR-10 data set

## Reference papers and blogs and others

\> ResNet
Deep Residual Learning for Image Recognition  

\> DenseNet  
Densely Connected Convolutional Networks  
https://sike6054.github.io/blog/paper/sixth-post/

\> PyramidNet  

\> Mixup

## Result

---

### __[ Default CNN ]__  

__Default CNN 2-layer__  
Accuracy : 53% (2 epoch)  
Accuracy : 62% (5 epoch)  
Accuracy : 63% (10 epoch)  
Accuracy : 62% (2 epoch with batch normalization)  
__Default CNN 3-layer__  
Accuracy : 54%  (2 epoch)  
Accuracy : 69% (5 epoch)  
Accuracy : 70% (10 epoch)  

### __[ ResNet ]__  

__ResNet-32__  
Accuracy : 91.8% ( with data augmentation, 80 epoch, 128 mini-batch, learning schedular )

### __[ DenseNet ]__  

__DenseNet (k=12, L=40)__  
Error : 8.68% (lr=0.01, 80 epoch) ( 5.24% in paper )  

__DenseNet-BC (k=12, L=100)__  
Error : 5.57% ( 4.51% in paper )  




