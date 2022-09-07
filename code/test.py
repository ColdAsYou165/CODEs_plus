'''
测试单独使用一个文件
'''
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
from torchvision.utils import save_image
# import torch.backends.cudnn as cudnn
import sys
from model import *
from utils import *
from models import resnet_orig

batch_size = 1024
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

trainloader_cifar10 = torch.utils.data.DataLoader(cifar10_train, batch_size=batch_size, shuffle=True, num_workers=2)
testloader_cifar10 = torch.utils.data.DataLoader(cifar10_test, batch_size=batch_size, shuffle=True, num_workers=2)
trainloader_cifar100 = torch.utils.data.DataLoader(cifar100_train, batch_size=batch_size, shuffle=True, num_workers=2)
testloader_cifar100 = torch.utils.data.DataLoader(cifar100_test, batch_size=batch_size, shuffle=True, num_workers=2)
trainloader_svhn = torch.utils.data.DataLoader(svhn_train, batch_size=batch_size, shuffle=True, num_workers=2)
testloader_svhn = torch.utils.data.DataLoader(svhn_test, batch_size=batch_size, shuffle=True, num_workers=2)

# 模型
model_resnet18 = resnet_orig.ResNet18(num_classes=10).cuda()


def test_resnet_acc_and_mmc(model=None, root=""):
    '''
    检查resnet模型的acc和mmc
    :return:
    '''
    # root = "/mnt/data/maxiaolong/CODEsSp/results/train_resnet_github/pth/resnet_github--acc0.9556.pth"
    # root = "/mnt/data/maxiaolong/CODEsSp/results/train_resnet_basline/pth/resnet18_baseline--epoch199--acc0.95.pth"
    print(f"{root}")
    model.load_state_dict(torch.load(root))
    acc_train = get_acc(model, trainloader_cifar10)
    acc_test = get_acc(model, testloader_cifar10)
    mmc_cifar10_train = get_mmc(model, trainloader_cifar10)
    mmc_cifar10_test = get_mmc(model, testloader_cifar10)
    mmc_cifar100_train = get_mmc(model, trainloader_cifar100)
    mmc_cifar100_test = get_mmc(model, testloader_cifar100)
    mmc_svhn_train = get_mmc(model, trainloader_svhn)
    mmc_svhn_test = get_mmc(model, testloader_svhn)
    print("acc", acc_train, acc_test)
    print("mmc_cifar10", mmc_cifar10_train, mmc_cifar10_test)
    print("mmc_cifar100", mmc_cifar100_train, mmc_cifar100_test)
    print("mmc_svhn", mmc_svhn_train, mmc_svhn_test)
    '''
    ../betterweights/resnet18_baseline_trainedbymiao_acc0.9532.pth
    acc 1.0 0.9532
    mmc_cifar10 0.9986795397949219 0.9780302856445312
    mmc_cifar100 0.8124723669433593 0.8228418884277344
    mmc_svhn 0.7096384452208385 0.789624792707428
    '''


if __name__ == "__main__":
    pass
    # test_resnet_acc_and_mmc(model=model_resnet18, root="../betterweights/resnet18_baseline_trainedbymiao_acc0.9532.pth")
