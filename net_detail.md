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

