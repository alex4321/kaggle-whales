import torch
from torch.autograd.variable import Variable
import torch.nn.functional as F

t = torch.FloatTensor(16,3,64,64).zero_() + 1e-10
t[-1, :, :, :] += 1e-1
a = Variable(t,requires_grad=True)

b = a.std(dim=0).mean()  # NaN gradients
#b = (a - a.mean(dim=0,keepdim=True)).norm(p=2,dim=0).mean() / (15**0.5)  # 0 gradients

b.backward()

print(a.grad)