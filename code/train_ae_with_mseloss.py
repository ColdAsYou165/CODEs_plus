'''
先训练一个能正常重构图像的ae
使用mse
'''
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.0006)
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--gpus", default="1")
args = parser.parse_args()
import os

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
from utils import *
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
# import torch.backends.cudnn as cudnn
import sys
from model import *

# 起势
name_args = get_args_str(args)
print(name_args)
num_classes = 10
batch_size = 128
epochs = args.epochs
lr = args.lr
# 不带v是有scheduler
# v1为去掉 scheduler,还真就去了效果更好
# v2 发现v1没训练完.在v1的基础上重新训练,增大epoch数,
name_project = "train_ae_with_mseloss_v2"
root_result = f"../results/{name_project}"
os.makedirs(root_result, exist_ok=True)
root_pth = root_result + "/pth"
root_pic = root_result + "/pic"
os.makedirs(root_pth, exist_ok=True)
os.makedirs(root_pic, exist_ok=True)
# [, 3, 32, 32]

# 数据集
cifar10_train = torchvision.datasets.CIFAR10(root="../data/cifar10", train=True, download=False,
                                             transform=transform_train_cifar_miao)
cifar10_test = torchvision.datasets.CIFAR10(root="../data/cifar10", train=False, download=False,
                                            transform=transform_test_cifar_miao)

trainloader_cifar10 = torch.utils.data.DataLoader(cifar10_train, batch_size=batch_size, shuffle=True, num_workers=2)
testloader_cifar10 = torch.utils.data.DataLoader(cifar10_test, batch_size=batch_size, shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
pic_origin, _ = next(iter(testloader_cifar10))
save_image(pic_origin, root_pic + "/origin.jpg")
pic_origin = pic_origin.cuda()
# 模型

model_g = model_g = AutoEncoder_Miao().cuda()
criterion_mse = torch.nn.MSELoss().cuda()
criterion_bce = torch.nn.BCEWithLogitsLoss().cuda()  # github上用的这个
model_g.apply(weights_init)
optimizer = torch.optim.Adam(model_g.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
# 训练
for epoch in range(epochs):
    model_g.train()
    loss_train, loss_test = 0, 0
    for batch_idx, (data, label) in enumerate(trainloader_cifar10):
        data = data.cuda()
        label = label.cuda()
        recontrust = model_g(data)
        loss = criterion_mse(recontrust, data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
    # scheduler.step()
    loss_train /= len(trainloader_cifar10)

    model_g.eval()
    for batch_idx, (data, label) in enumerate(testloader_cifar10):
        with torch.no_grad():
            data = data.cuda()
            label = label.cuda()
            recontrust = model_g(data)
            loss = criterion_mse(recontrust, data)
            loss_test += loss.item()
    loss_test /= len(testloader_cifar10)

    pic_reconstruct = model_g(pic_origin)
    save_image(pic_reconstruct, root_pic + f"/reconstruct--epoch{epoch}--loss{loss_train}.jpg")
    print(f"[{epoch}/{epochs}] : loss_train=", loss_train, "loss_test=", loss_test)
    if True and epoch > 180:
        torch.save(model_g.state_dict(), root_pth + f"/ae_miao_trainedbybclloss--epoch{epoch}--loss{loss_test}.pth")
