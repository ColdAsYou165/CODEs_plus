import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# from model import AutoEncoder_Miao
import torch
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from model import AutoEncoder_Miao
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
'''a=13
a=int(a*0.5*2)
print(a)'''
num_classes = 10
batch_size = 4
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
trainset = torchvision.datasets.CIFAR10(root="../data/cifar10", train=True, download=False,
                                        transform=transform_train_cifar_miao)
testset = torchvision.datasets.CIFAR10(root="../data/cifar10", train=False, download=False,
                                       transform=transform_test_cifar_miao)
trainloader = DataLoader(trainset, batch_size, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size, shuffle=True, num_workers=2)

model = AutoEncoder_Miao().cuda()

for data, label in trainloader:
    data = data.cuda()
    label = label.cuda()
    virtual_data, virtual_label = model.generate_virtual(data, label, set_encoded_detach=True, train_generate=True,
                                                         num_classes=num_classes, scale_generate=1)
    print(virtual_data.shape, virtual_label.shape)
    print(label)
    print("---")
    print(virtual_label)
