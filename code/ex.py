import datetime
import os
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--gpus", default="0")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
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

'''num_classes = 10
batch_size = 128
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
trainset = torchvision.datasets.CIFAR10(root="../data/cifar10", train=True, download=False,
                                        transform=transform_train_cifar_miao)
testset = torchvision.datasets.CIFAR10(root="../data/cifar10", train=False, download=False,
                                       transform=transform_test_cifar_miao)
trainloader = DataLoader(trainset, batch_size, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size, shuffle=True, num_workers=2)

model = AutoEncoder_Miao().cuda()
criterion = torch.nn.MSELoss()
for data, label in trainloader:
    data = data.cuda()
    label = label.cuda()
    virtual_data, virtual_label = model.generate_virtual(data, label, True, True)
    loss = criterion(virtual_data, data)
    loss.backward()
    print(data.requires_grad)
    print(virtual_data.requires_grad, virtual_label.requires_grad)'''
'''a=torch.randn([4,3,28,28])
b=a.repeat_interleave(3,dim=0)
print(b.shape)'''
'''for i in range(10):
    acc = i
    mmc_cifar10 = i + 1
    mmc_cifar100 = i + 2
    mmc_svhn = i + 3
    with open("./mmc.txt", "a+") as results_file:
        results_file.write(f"acc={acc},mmc_cifar10={mmc_cifar10},mmc_cifar100={mmc_cifar100},mmc_svhn={mmc_svhn}\r\n")'''
from model import AutoEncoder_Miao

'''
model = AutoEncoder_Miao()
data = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
label = torch.tensor([0, 1, 2, 3, 1, 2, 3])
model.generate_virtual_v1(data, label, True, 4)'''
'''x=torch.tensor([0,1,0,0]).float()
y=torch.tensor([0,0,0,0]).float()
criterion_cel=nn.CrossEntropyLoss()
loss=criterion_cel(x,y)
print(loss+1)'''
'''result = torch.zeros_like(data)
print(result)
index_ji = range(1, len(result), 2)
index_ou = range(0, len(result), 2)
result[index_ji] = data[index_ou]
result[index_ou]=data[index_ji]
print(result)'''
# 反转卷积
# data=torch.tensor([[0,1,2,3],[4,5,6,7],[0,1,2,3]])
# print(data[0]==data[2])
'''from datetime import datetime

a=datetime.now().strftime('%y-%m-%d,%H-%M-%S')
print(a)
print(type(a))'''
x=[[i,2*i] for i in range(10)]
x=torch.te