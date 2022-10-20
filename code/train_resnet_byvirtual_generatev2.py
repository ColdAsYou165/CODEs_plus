'''
文件目的 : 查看压制训练效果
生成虚假样本的方法 : 按照权重将两个样本的特征向量加到一起再decoder
描述 : 与genneratev2配套
'''
import os
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=200, help="这个就不调整了,就默认200")
parser.add_argument("--gpus", default="0")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--loss_virtual_weight", type=float, default=1, help="压制训练时候,loss_virtual的权重")
# v5版本专用 选择ae参数
parser.add_argument("--ae_version", type=int, default=0, help="ae权重版本,2为1e-8  3为1e-4 4为1e-5")
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
from utils import *

# 起势
print(str(args))
name_runs = f"runs/yazhixunlian_genev2--ae_version{args.ae_version}--loss_weight{args.loss_virtual_weight}"
writer = SummaryWriter(name_runs)
writer.add_text("args", str(args))
name_args = get_args_str(args)
name_project = "train_resnet18_bygenev2"
results_root = f"../results/{name_project}" + f"/{name_args}"
os.makedirs(results_root, exist_ok=True)
results_root_pic = results_root + "/pic"
os.makedirs(results_root_pic, exist_ok=True)
results_root_pth = results_root + "/pth"
os.makedirs(results_root_pth, exist_ok=True)

num_classes = 10
# 模型
model_d = resnet_orig.ResNet18(num_classes=num_classes).cuda()
model_d.apply(weights_init)

model_g = AutoEncoder_Miao().cuda()
#
if args.ae_version == 0:
    # closs+wloss+bloss1e-4
    root_ae = "/mnt/data/maxiaolong/CODEsSp/results/train_ae_with_3loss_generatev2_v1/argsbatch_size128--beta10.5--blend_loss_weight0.0001--epochs2000--gpus'4'--lr_dis6e-05--lr_g6e-05--lr_scale10000.0--optimizer'Adam'--w_loss_weight1e-05/pth/model_chamfer_and_wloss--epoch1999.pth"
    state_g = torch.load(root_ae)["model"]
elif args.ae_version == 1:
    # closs+wloss+bloss1e-5
    root_ae = "/mnt/data/maxiaolong/CODEsSp/results/train_ae_with_3loss_generatev2_v1/argsbatch_size128--beta10.5--blend_loss_weight1e-05--epochs2000--gpus'1'--lr_dis6e-05--lr_g6e-05--lr_scale10000.0--optimizer'Adam'--w_loss_weight1e-05/pth/model_chamfer_and_wloss--epoch1999.pth"
    state_g = torch.load(root_ae)["model"]
elif args.ae_version == 2:
    # closs+wloss+bloss1e-6
    root_ae = "/mnt/data/maxiaolong/CODEsSp/results/train_ae_with_3loss_generatev2_v1/argsbatch_size128--beta10.5--blend_loss_weight1e-06--epochs2000--gpus'5'--lr_dis6e-05--lr_g6e-05--lr_scale10000.0--optimizer'Adam'--w_loss_weight1e-05/pth/model_chamfer_and_wloss--epoch1999.pth"
    state_g = torch.load(root_ae)["model"]
elif args.ae_version == 3:
    # wloss+bloss1
    root_ae = "/mnt/data/maxiaolong/CODEsSp/results/train_ae_with_3loss_generatev2_v2/argslr_g6e-05--lr_dis6e-05--lr_scale10000.0--optimizer'Adam'--epochs2000--gpus'5'--batch_size128--beta10.5--w_loss_weight1e-05--blend_loss_weight1/pth/model_chamfer_and_wloss--epoch1999.pth"
    state_g = torch.load(root_ae)["model"]
model_g.load_state_dict(state_g)
model_g.eval()
# optimizer_d = torch.optim.Adam(model_d.parameters(), lr=args.lr)#

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
print(f"压制训练前:acc:{acc},mmc_cifar100={mmc_cifar100},mmc_svhn={mmc_svhn},mmc_cifar10={mmc_cifar10}")

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
        data_virtual, label_virtual = model_g.generate_virtual_v2(data, label, set_encoded_detach=True,
                                                                  train_generate=False, num_classes=num_classes)
        data_virtual = data_virtual.detach()
        pred_normal = model_d(data_normal)
        loss_normal = criterion(pred_normal, label_normal)
        pred_virtual = model_d(data_virtual)
        loss_virtual = criterion(pred_virtual, label_virtual)
        loss = (loss_virtual + loss_normal).mean()
        optimizer_d.zero_grad()
        loss_normal.backward()
        (args.loss_virtual_weight * loss_virtual).backward()
        optimizer_d.step()
        loss_train_containv += loss.item()
        # 没啥用就是看一下ae生成看来咋样的虚假图像用于压制训练
        if batch_idx == 0 and epoch % 10 == 0:
            pic = torch.concat([data, data_virtual], dim=0)
            save_image(pic, results_root_pic + f"/virtualpic--epoch{epoch}.jpg")

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
                   f"{results_root}/resnet18_yazhixunlian__acc{acc:.2f}__cimmc{mmc_cifar100:.2f}__svhnmmc{mmc_svhn}.pth")
    writer.add_scalars("mmc", {"mmc_cifar10": mmc_cifar10, "mmc_cifar100": mmc_cifar100,
                               "mmc_svhn": mmc_svhn}, epoch + 1)
    writer.add_scalar("cifar10_acc", acc, epoch + 1)
    writer.add_scalar("train_containv", loss_train_containv, epoch + 1)

    print(f"epoch[{epoch}/{args.epochs}] : cifar10_test_acc={acc} , ", "mmc_test_cifar10=", mmc_cifar10)
    print("loss_train_containv=", loss_train_containv, "mmc_cifar100=", mmc_cifar100, " , mmc_svhn=", mmc_svhn)
    print("-" * 40)
with open("./runs/new_mmc.txt", "a+") as results_file:
    results_file.write(
        f"--lossweight{args.loss_virtual_weight}--ae_version{args.ae_version} : acc={acc},mmc_cifar10={mmc_cifar10},mmc_cifar100={mmc_cifar100},mmc_svhn={mmc_svhn}\r\n")
