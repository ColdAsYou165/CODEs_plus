'''
CEDA 就是单纯用噪声进行压制训练
'''
import os
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=200, help="这个就不调整了,就默认200")
parser.add_argument("--gpus", default="0")
parser.add_argument("--batch_size", type=int, default=128)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
from torchvision.utils import save_image
# import torch.backends.cudnn as cudnn
import sys
from model import *
from models import resnet_orig
from utils import *

# 起势
print(str(args))
name_runs = f"runs/CEDA{datetime.now()}"
writer = SummaryWriter(name_runs)
writer.add_text("args", str(args))
name_args = get_args_str(args)
name_project = "train_resnet18_byCEDA"
results_root = f"../results/{name_project}" + f"/{name_args}"
os.makedirs(results_root, exist_ok=True)
results_root_pic = results_root + "/pic"
os.makedirs(results_root_pic, exist_ok=True)
results_root_pth = results_root + "/pth"
os.makedirs(results_root_pth, exist_ok=True)
# 数据集
cifar10_train = torchvision.datasets.CIFAR10(root="../data/cifar10", train=True, download=False,
                                             transform=transform_train_cifar_miao)
cifar10_test = torchvision.datasets.CIFAR10(root="../data/cifar10", train=False, download=False,
                                            transform=transform_test_cifar_miao)
cifar100_train = torchvision.datasets.CIFAR100(root="../data/cifar100", train=True, download=False,
                                               transform=transform_train_cifar_miao)
cifar100_test = torchvision.datasets.CIFAR100(root="../data/cifar100", train=False, download=False,
                                              transform=transform_test_cifar_miao)
svhn_train = torchvision.datasets.SVHN(root="../data/svhn", split="train", download=False,
                                       transform=transform_train_cifar_miao)
svhn_test = torchvision.datasets.SVHN(root="../data/svhn", split="test", download=False,
                                      transform=transform_test_cifar_miao)

trainloader_cifar10 = torch.utils.data.DataLoader(cifar10_train, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=2)
testloader_cifar10 = torch.utils.data.DataLoader(cifar10_test, batch_size=args.batch_size, shuffle=True, num_workers=2)
trainloader_cifar100 = torch.utils.data.DataLoader(cifar100_train, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=2)
testloader_cifar100 = torch.utils.data.DataLoader(cifar100_test, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=2)
trainloader_svhn = torch.utils.data.DataLoader(svhn_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
testloader_svhn = torch.utils.data.DataLoader(svhn_test, batch_size=args.batch_size, shuffle=True, num_workers=2)
num_classes = 10
lr = 0.1
# 模型
lam = 1.
acc_std = 0.9
loader_noise = None
model_d = resnet_orig.ResNet18(num_classes=num_classes).cuda()
model_d.apply(weights_init)
# criterion = nn.NLLLoss()
criterion = nn.CrossEntropyLoss()
optimizer_d = torch.optim.SGD(model_d.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)  # 苗lr0.1
for epoch in range(args.epochs):
    if epoch >= 60:
        lr = 2e-2
    if epoch >= 120:
        lr = 4e-3
    if epoch >= 160:
        lr = 8e-4
    adjust_learning_rate(optimizer_d, lr)

    model_d.train()
    loss_train = 0
    acc_train = 0
    # p_in = torch.tensor(1. / (1. + lam)).float().cuda()
    # p_out = torch.tensor(lam / (1. + lam)).float().cuda()
    p_in = torch.tensor(1.).float().cuda()
    p_out = torch.tensor(1.).float().cuda()
    if loader_noise is not None:
        enum = enumerate(loader_noise)
    for batch_idx, (data, label) in enumerate(trainloader_cifar10):
        data = data.cuda()
        label = label.cuda()
        noise = torch.rand_like(data)
        if loader_noise is not None:
            noise = enum.__next__()[1][0].cuda()

        full_data = torch.concat([data, noise], dim=0)
        full_out = model_d(full_data)
        output = full_out[:data.shape[0]]
        output_adv = full_out[data.shape[0]:]
        loss_1 = criterion(output, label)
        laebl_adv = torch.ones([data.shape[0], num_classes]).cuda() * 0.1
        loss_2 = criterion(output_adv, laebl_adv)
        # loss_2 = -output_adv.mean()
        loss = p_in * loss_1 + p_out * loss_2
        # print(loss_1.item(), loss_2.item(), loss.item())
        optimizer_d.zero_grad()
        loss.backward()
        optimizer_d.step()

        loss_train += loss.item()
        _, predicted = output.max(1)
        acc_train += predicted.eq(label).sum().item()
    loss_train /= len(trainloader_cifar10)
    acc_train /= len(trainloader_cifar10.dataset)

    print(f"---------\n[{epoch}/{args.epochs}]")
    mmc_cifar100 = get_mmc(model_d, testloader_cifar100)
    mmc_svhn = get_mmc(model_d, testloader_svhn)
    mmc_cifar10 = get_mmc(model_d, testloader_cifar10)
    acc = get_acc(model_d, testloader_cifar10)
    print(f"acc = {acc},mmc_cifar10 = {mmc_cifar10}")
    print(f"mmc_cifar100 = {mmc_cifar100} , mmc_svhn = {mmc_svhn}")

    if acc > acc_std:
        acc_std = acc
        state_d = {"model": model_d.state_dict()}
        torch.save(state_d,
                   f"{results_root_pth}/resnet18_yazhixunlian__acc{acc:.2f}__cimmc{mmc_cifar100:.2f}__svhnmmc{mmc_svhn:.2f}.pth")
