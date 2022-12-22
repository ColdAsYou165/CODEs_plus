'''
文件目的 : 查看压制训练效果
生成虚假样本的方法 : 按照权重将两个样本的特征向量加到一起,label同样方法加到一起,encoded和label拼到一起,再decoder
描述 : 与train_ae_containy_withtangloss.py配套
'''
import os
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=200, help="这个就不调整了,就默认200")
parser.add_argument("--gpus", default="5")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--loss_virtual_weight", type=float, default=1, help="压制训练时候,loss_virtual的权重")
parser.add_argument("--ae_version", type=int, default=4, help="ae权重版本")
parser.add_argument("--scale", type=int, default=4)
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
from torchvision.utils import save_image
# import torch.backends.cudnn as cudnn
import sys
from model import *
from models import resnet_orig
from models.ae_miao_chromosome import *
from utils import *

# /v3新加单个类别tyangloss训练的
# v4随便试
name_project = "train_resnet_by_virtual_generated_by_aetrainedbychromosomev5_83"
args_str = get_args_str(args)
root_result, (root_pth, root_pic) = getResultDir(name_project=name_project, name_args=args_str)
log = getLogger(formatter_str=None, root_filehandler=root_result + f"/logger.log")
log.info(str(args))
seed = random.randint(0, 2022)
setup_seed(seed)
log.info(f"seed={seed}")

# 起势
name_runs = f"{root_result}" + "/runs"
os.makedirs(name_runs, exist_ok=True)
writer = SummaryWriter(name_runs)
writer.add_text("args", str(args))

num_classes = 10
# 模型
model_d = resnet_orig.ResNet18(num_classes=num_classes).cuda()
model_d.apply(weights_init)

model_g = AutoEncoder_Miao_crossover(num_classes).cuda()
#
if args.ae_version == 0:
    root_ae = "../weights_ex/ae_chromsome_suibianshi/ae_generatevirtual--epoch399.pth"
    state_g = torch.load(root_ae)
elif args.ae_version == 1:
    root_ae = "../weights_ex/ae_chromsome_suibianshi/ae_generatevirtual--epoch599.pth"
    state_g = torch.load(root_ae)
elif args.ae_version == 2:
    root_ae = "../weights_ex/ae_chromsome_suibianshi/ae_generatevirtual--epoch799.pth"
    state_g = torch.load(root_ae)
elif args.ae_version == 3:
    root_ae = "../weights_ex/ae_chromsome_suibianshi/ae_generatevirtual--epoch1199.pth"
    state_g = torch.load(root_ae)
elif args.ae_version == 4:
    root_ae = "../weights_ex/ae_chromsome_suibianshi/ae_generatevirtual--epoch1599.pth"
    state_g = torch.load(root_ae)
elif args.ae_version == 5:
    root_ae = "../weights_ex/ae_chromsome_suibianshi/ae_generatevirtual--epoch1799.pth"
    state_g = torch.load(root_ae)
model_g.load_state_dict(state_g)
model_g.eval()

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

# 优化器
criterion = torch.nn.CrossEntropyLoss().cuda()
lr = 1e-1
optimizer_d = torch.optim.SGD(model_d.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)  # 苗lr0.1
# 记录起初的mmc
mmc_cifar10 = get_mmc(model_d, testloader_cifar10)
mmc_cifar100 = get_mmc(model_d, testloader_cifar100)
mmc_svhn = get_mmc(model_d, testloader_svhn)
acc = get_acc(model_d, testloader_cifar10)
writer.add_scalars("mmc", {"mmc_cifar100": mmc_cifar100, "mmc_svhn": mmc_svhn, "mmc_cifar10": mmc_cifar10}, 0)
writer.add_scalar("cifar10_acc", acc, 0)
log.info(f"压制训练前:acc:{acc},mmc_cifar100={mmc_cifar100},mmc_svhn={mmc_svhn},mmc_cifar10={mmc_cifar10}")

# 压制训练
acc_std = 0.80
num_classes = 10
for epoch in range(args.epochs):
    # lr 2e-2 提高的多 但是只能到0.86
    # 苗调整学习率
    if epoch >= 60:
        lr = 2e-2
    if epoch >= 120:
        lr = 4e-3
    if epoch >= 160:
        lr = 8e-4
    adjust_learning_rate(optimizer_d, lr)

    # 训练
    model_d.train()
    loss_train_containv = 0
    for batch_idx, (data, label) in enumerate(trainloader_cifar10):
        data = data.cuda()
        label = label.cuda()
        data_normal = data.detach()
        label_normal = F.one_hot(label, num_classes).detach().float()
        # data_virtual = model_g.module.generate_virtual(data).detach()
        # 压制训练时候,虚假样本的label应该都是0.1,我设置错了.
        data_virtual, label_virtual = model_g.generate_virtual_anquanzhong(data, label, set_differentlabel=True,
                                                                           set_virtuallabel_uniform=True, scale=args.scale)
        # 两行代码完成mixup
        index_mixup = torch.randperm(len(data_virtual))
        data_virtual = 0.5 * data_virtual + 0.5 * data_virtual[index_mixup]

        data_virtual = data_virtual.detach()
        pred_normal = model_d(data_normal)
        loss_normal = criterion(pred_normal, label_normal)
        pred_virtual = model_d(data_virtual)
        loss_virtual = criterion(pred_virtual, label_virtual)
        loss = (loss_virtual + loss_normal).mean()
        optimizer_d.zero_grad()
        (loss_normal + args.loss_virtual_weight * loss_virtual).backward()
        optimizer_d.step()
        loss_train_containv += loss.item()
        # 没啥用就是看一下ae生成看来咋样的虚假图像用于压制训练
        if batch_idx == 0 and epoch % 10 == 0:
            pic = torch.concat([data, data_virtual], dim=0)
            save_image(pic, root_pic + f"/virtualpic--epoch{epoch}.jpg")

    loss_train_containv /= len(trainloader_cifar10)
    # 测试
    mmc_cifar100 = get_mmc(model_d, testloader_cifar100)
    mmc_svhn = get_mmc(model_d, testloader_svhn)
    mmc_cifar10 = get_mmc(model_d, testloader_cifar10)
    acc = get_acc(model_d, testloader_cifar10)
    if acc > acc_std:
        acc_std = acc
        state_d = {"model": model_d.state_dict()}
        torch.save(state_d,
                   f"{root_pth}/resnet18_yazhixunlian__acc{acc:.2f}__cimmc{mmc_cifar100:.2f}__svhnmmc{mmc_svhn}.pth")
    writer.add_scalars("mmc", {"mmc_cifar10": mmc_cifar10, "mmc_cifar100": mmc_cifar100,
                               "mmc_svhn": mmc_svhn}, epoch + 1)
    writer.add_scalar("cifar10_acc", acc, epoch + 1)
    writer.add_scalar("train_containv", loss_train_containv, epoch + 1)

    log.info(f"epoch[{epoch}/{args.epochs}] : cifar10_test_acc={acc},mmc_test_cifar10={mmc_cifar10}")
    log.info(f"\tloss_train_containv={loss_train_containv},mmc_cifar100={mmc_cifar100},mmc_svhn={mmc_svhn}")
    log.info(f"loss_normal={loss_normal},loss_virtual={loss_virtual}")
    log.info("-" * 40)
with open(f"../results/{name_project}/{name_project}.txt", "a+") as results_file:
    # results_file.write(f"{args}\r\n")
    results_file.write(
        f"--lossweight{args.loss_virtual_weight}--ae_version{args.ae_version} : acc={acc},mmc_cifar10={mmc_cifar10},mmc_cifar100={mmc_cifar100},mmc_svhn={mmc_svhn}\r\n")
