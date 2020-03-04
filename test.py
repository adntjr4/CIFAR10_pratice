import torch.nn.functional as F
import torch

x = torch.randn(1,1,2,2)
print(x)
out = F.pad(x, (2, 2, 0, 0, 0, 0))
print(out)
