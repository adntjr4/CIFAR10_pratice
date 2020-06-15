# Network implementation detail

## MobileNet

Same form of MobileNet Body Architecture table for CIFAR-10 instead of ImageNet

| Type / Stride | Filter Shape | Input Size |
|---|---|---|
| Conv / s1 | 3 x 3 x 3 x 32 | 32 x 32 x 3 |
| Conv dw / s1 | 3 x 3 x 32 dw | 32 x 32 x 32 |
| Conv s1 | 1 x 1 x 32 x 64 | 32 x 32 x 32 |
| Conv dw / s2 | 3 x 3 x 64 dw | 32 x 32 x 64 |
| Conv s1 | 1 x 1 x 64 x 128 | 16 x 16 x 64 |
| Conv dw / s1 | 3 x 3 x 128 dw | 16 x 16 x 128 |
| Conv s1 | 1 x 1 x 128 x 128 | 16 x 16 x 128 |
| Conv dw / s2 | 3 x 3 x 128 dw | 16 x 16 x 128 |
| Conv s1 | 1 x 1 x 128 x 256 | 8 x 8 x 128 |
| Conv dw / s1 | 3 x 3 x 256 dw | 8 x 8 x 256 |
| Conv s1 | 1 x 1 x 256 x 256 | 8 x 8 x 256 |
| Conv dw / s2 | 3 x 3 x 256 dw | 8 x 8 x 256 |
| Conv s1 | 1 x 1 x 256 x 512 | 4 x 4 x 256 |
| 5x Conv dw / s1 <br> 5x Conv / s1 | 3 x 3 x 512 dw <br> 1 x 1 x 512 x 512 | 4 x 4 x 512 <br> 4 x 4 x 512 |
| Avg Pool / s1 | Pool 4 x 4 | 4 x 4 x 512 |
| FC / s1 | 512 x 10 | 1 x 1 x 512 |
| Softmax / s1 | Classifier | 1 x 1 x 10 |

## MobileNetV2

Same form of MobileNetV2 Body Architecture table for CIFAR-10 instead of ImageNet

| Input | Operator | t | c | n | s |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 32 x 32 x 3 | conv2d | - | 32 | 1 | 1 |
| 32 x 32 x 32 | bottleneck | 1 | 16 | 1 | 1 |
| 32 x 32 x 16 | bottleneck | 6 | 24 | 2 | 2 |
| 16 x 16 x 24 | bottleneck | 6 | 32 | 3 | 2 |
| 8 x 8 x 32 | bottleneck | 6 | 64 | 4 | 2 |
| 4 x 4 x 64 | bottleneck | 6 | 96 | 3 | 1 |
| 4 x 4 x 96 | bottleneck | 6 | 160 | 3 | 2 |
| 2 x 2 x 160 | bottleneck | 6 | 320 | 1 | 1 |
| 2 x 2 x 320 | conv2d 1x1 | - | 1280 | 1 | 1 |
| 2 x 2 x 1280 | avgpool 2x2 | - | - | 1 | - |
| 1 x 1 x 1280 | conv2d 1x1 | - | k | - |  |