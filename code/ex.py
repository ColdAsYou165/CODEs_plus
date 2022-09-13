import os
# from model import AutoEncoder_Miao
import torch
import numpy as np
import torchvision
from torch.utils.data import DataLoader

from utils import *

'''x = torch.range(1, 16).reshape([4, 4])
print(x)
idx = torch.randperm(x.shape[0])
y = x[idx]
print(y)
x += 1
print(x)
print(y)'''
'''a=torch.tensor([[0,1,1],
                [1,0,1],
                [0,2.,0],
                [0,1,0],
                [2,0,0]
                ])
c=torch.where(a==2)[0].numpy()
c=list(c)
print(c)
idx=np.arange(0,a.shape[0])
print(idx)
i=np.setdiff1d(idx,c)
print(i)
a=a[i]
print(a[:10])'''
a=13
a=int(a*0.5*2)
print(a)