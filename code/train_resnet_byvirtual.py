'''
描述: 使用训练好的ae压制训练
---
说明:
    老师说,兄严格按照苗师代码来.
    epochs200 lr0.1以及后续调整策略 优化器
'''
import os
import time
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument("--method", default="")
parser.add_argument("--epochs", type=int, default=200, help="这个就不调整了,就默认200")
parser.add_argument("--gpus", default="0")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--loss_virtual_weight", type=float, default=1, help="压制训练时候,loss_virtual的权重")
# v5版本专用 选择ae参数
parser.add_argument("--ae_version", type=int, default=16, help="ae权重版本,2为1e-8  3为1e-4 4为1e-5")
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
name_runs = f"runs/oppress--ae_version{args.ae_version}--loss_weight{args.loss_virtual_weight}"
writer = SummaryWriter(name_runs)
writer.add_text("args", str(args))
# v2观察 model567序号的 压制与效果
# v4 使用的ae为containy的ae,训练1800轮,但是有时不能重构图像
name_project = "train_resnet18_byvirtual_v4"
name_args = get_args_str(args)
results_root, (results_pth_root, results_pic_root) = getResultDir(name_project, name_args, results_root="../results")

num_classes = 10
# 模型
model_d = resnet_orig.ResNet18(num_classes=num_classes).cuda()
model_d.apply(weights_init)

if args.ae_version < 12:
    model_g = AutoEncoder_Miao().cuda()
elif args.ae_version > 11:
    model_g = AutoEncoder_Miao_containy(num_classes).cuda()
# model_g.apply(weights_init)
# v5的state_g
# 0,1为之前的两个,当时压制训练有一点效果,但是现在怀疑是那10%生成的正常样本带来的影响
# 2为1e-8  3为1e-4 4为1e-5
# 8为weight 1e-5 epoch 1199
# 9 10 11 分别为ae--epoch1199 blendweight分别为 1e-5颜色正常 1e-5颜色深 1e-6
# 12 13 14为 containy的ae,训练ae时候的blendweight分别为 1 1e-3 1e-5
# 15 containy的ae,训练ae的时候blendweight为1,lr2e-5,观察batch能生成正常的虚假图像
if args.ae_version == 0:
    state_g = torch.load(
        f"../betterweights/ae_trained_by3loss_v0/ae_chamfer_and_wloss_and_crossloss--crossweight1e-4_epoch799.pth")[
        "model"]
elif args.ae_version == 1:
    state_g = torch.load(
        f"../betterweights/ae_trained_by3loss_v0/ae_chamfer_and_wloss_and_crossloss--crossweight1e-5_epoch799.pth")[
        "model"]
elif args.ae_version == 2:
    state_g = torch.load(
        "/mnt/data/maxiaolong/CODEsSp/results/train_ae_with3loss_chamfer_blend_w_gaijincross_v2/label0.9-0.1argslr_g0.0002--lr_dis0.0002--lr_scale10000.0--optimizer'Adam'--epochs500--gpus'7'--batch_size128--beta10.5--w_loss_weight1e-05--blend_loss_weight1e-08/pth/model_chamfer_and_wloss--epoch299.pth")[
        "model"]
elif args.ae_version == 3:
    state_g = torch.load(
        "/mnt/data/maxiaolong/CODEsSp/results/train_ae_with3loss_chamfer_blend_w_gaijincross_v2/label0.9-0.1argslr_g0.0002--lr_dis0.0002--lr_scale10000.0--optimizer'Adam'--epochs500--gpus'3'--batch_size128--beta10.5--w_loss_weight1e-05--blend_loss_weight0.0001/pth/model_chamfer_and_wloss--epoch299.pth")[
        "model"]
elif args.ae_version == 4:
    state_g = torch.load(
        "/mnt/data/maxiaolong/CODEsSp/results/train_ae_with3loss_chamfer_blend_w_gaijincross_v2/label0.9-0.1argslr_g0.0002--lr_dis0.0002--lr_scale10000.0--optimizer'Adam'--epochs500--gpus'4'--batch_size128--beta10.5--w_loss_weight1e-05--blend_loss_weight1e-05/pth/model_chamfer_and_wloss--epoch299.pth")[
        "model"]
elif args.ae_version == 6:
    # 699轮 blendlossweight为1e-6
    state_g = torch.load(
        "/mnt/data/maxiaolong/CODEsSp/results/train_ae_with3loss_chamfer_blend_w_gaijincross_v4/argslr_g0.0002--lr_dis0.0002--lr_scale10000.0--optimizer'Adam'--epochs1800--gpus'4'--batch_size128--beta10.5--w_loss_weight1e-05--blend_loss_weight1e-06/pth/model_chamfer_and_wloss--epoch699.pth")[
        "model"]
elif args.ae_version == 5:
    # 699轮 blendlossweight为1e-5
    state_g = torch.load(
        "/mnt/data/maxiaolong/CODEsSp/results/train_ae_with3loss_chamfer_blend_w_gaijincross_v4/argslr_g0.0002--lr_dis0.0002--lr_scale10000.0--optimizer'Adam'--epochs1800--gpus'1'--batch_size128--beta10.5--w_loss_weight1e-05--blend_loss_weight1e-05/pth/model_chamfer_and_wloss--epoch699.pth")[
        "model"]
elif args.ae_version == 7:
    # 199轮 blendlossweight为1e-5
    state_g = torch.load(
        "/mnt/data/maxiaolong/CODEsSp/results/train_ae_with3loss_chamfer_blend_w_gaijincross_v4/argslr_g0.0002--lr_dis0.0002--lr_scale10000.0--optimizer'Adam'--epochs1800--gpus'1'--batch_size128--beta10.5--w_loss_weight1e-05--blend_loss_weight1e-05/pth/model_chamfer_and_wloss--epoch199.pth")[
        "model"]
elif args.ae_version == 8:
    state_g = torch.load(
        "/mnt/data/maxiaolong/CODEsSp/results/train_ae_with3loss_chamfer_blend_w_gaijincross_v4/argslr_g0.0002--lr_dis0.0002--lr_scale10000.0--optimizer'Adam'--epochs1800--gpus'1'--batch_size128--beta10.5--w_loss_weight1e-05--blend_loss_weight1e-05/pth/model_chamfer_and_wloss--epoch1199.pth")[
        "model"]
elif args.ae_version == 9:
    state_g = \
        torch.load("../betterweights/ae_trainedby3loss/ae_miao_3loss_chamfer__w_blend1e-5--epoch1199--yansenormal.pth")[
            "model"]
elif args.ae_version == 10:
    state_g = \
        torch.load("../betterweights/ae_trainedby3loss/ae_miao_3loss_chamfer__w_blend1e-5--epoch1199--yanseshen.pth")[
            "model"]
elif args.ae_version == 11:
    state_g = torch.load("../betterweights/ae_trainedby3loss/ae_miao_3loss_chamfer__w_blend1e-6--epoch1199.pth")[
        "model"]
elif args.ae_version == 12:
    state_g = torch.load(
        "/mnt/data/maxiaolong/CODEsSp/results/ae_containy_generatevirtual_v2_w1/argsbatch_size128--beta10.5--blend_loss_weight1.0--epochs1800--gpus'0'--lr6e-05--lr_dis6e-05--method'train_generate_virtual'--w_loss_weight1/pthae_generatevirtual--epoch1799.pth")
elif args.ae_version == 13:
    state_g = torch.load(
        "/mnt/data/maxiaolong/CODEsSp/results/ae_containy_generatevirtual_v2_w1/argsbatch_size128--beta10.5--blend_loss_weight0.001--epochs1800--gpus'1'--lr6e-05--lr_dis6e-05--method'train_generate_virtual'--w_loss_weight1/pthae_generatevirtual--epoch1799.pth")
elif args.ae_version == 14:
    state_g = torch.load(
        "/mnt/data/maxiaolong/CODEsSp/results/ae_containy_generatevirtual_v2_w1/argsbatch_size128--beta10.5--blend_loss_weight1e-05--epochs1800--gpus'2'--lr6e-05--lr_dis6e-05--method'train_generate_virtual'--w_loss_weight1/pthae_generatevirtual--epoch1799.pth")
elif args.ae_version == 15:
    state_g = torch.load("../betterweights/train_ae_containy/ae_generatevirtual--epoch999.pth")
elif args.ae_version == 16:
    state_g = torch.load("../betterweights/train_ae_containy/caixvjian.pth")
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
criterion_cross = torch.nn.CrossEntropyLoss().cuda()
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
    loss_virtual_all = 0
    loss_normal_all = 0
    for batch_idx, (data, label) in enumerate(trainloader_cifar10):
        data = data.cuda()
        label = label.cuda()
        data_normal = data.detach()
        label_normal = F.one_hot(label, num_classes).detach().float()
        # data_virtual = model_g.module.generate_virtual(data).detach()
        data_virtual, label_virtual = model_g.generate_virtual(data, label, set_encode_detach=True,
                                                               set_virtual_label_uniform=True)
        pred_normal = model_d(data_normal)
        loss_normal = criterion_cross(pred_normal, label_normal)
        loss_normal_all += loss_normal.item()
        pred_virtual = model_d(data_virtual)
        loss_virtual = criterion_cross(pred_virtual, label_virtual)
        loss_virtual_all += loss_virtual.item()
        loss = (loss_virtual + loss_normal).mean()
        optimizer_d.zero_grad()
        loss_normal.backward()
        (args.loss_virtual_weight * loss_virtual).backward()
        optimizer_d.step()
        loss_train_containv += loss.item()

        #
        # 没啥用就是看一下ae生成看来咋样的虚假图像用于压制训练
        if True and epoch % 40 == 0:
            pic = torch.concat([data, data_virtual], dim=0)
            save_image(pic, results_pic_root + f"/virtualpic--epoch{epoch}--{batch_idx}.jpg")

    loss_train_containv /= len(trainloader_cifar10)
    loss_virtual_all /= len(trainloader_cifar10)
    loss_normal_all /= len(trainloader_cifar10)
    # 测试
    if True:
        # 保存好的resnet权重,输出压制训练需要观察的mmc acc的值
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
        writer.add_scalars("loss", {"loss_norm": loss_normal_all, "loss_virtual": loss_virtual_all}, epoch)
        print(f"[{epoch}/{args.epochs}]:loss_norm={loss_normal_all},loss_vir={loss_virtual_all}")
        print(f"epoch[{epoch}/{args.epochs}] : cifar10_test_acc={acc} , ", "mmc_test_cifar10=", mmc_cifar10)
        print("loss_train_containv=", loss_train_containv, "mmc_cifar100=", mmc_cifar100, " , mmc_svhn=", mmc_svhn)
        print("-" * 40)
with open("./runs/mmc_ae_containy.txt", "a+") as results_file:
    results_file.write(
        f"--lossweight{args.loss_virtual_weight},--ae_version{args.ae_version} : ,acc={acc},mmc_cifar10={mmc_cifar10},mmc_cifar100={mmc_cifar100},mmc_svhn={mmc_svhn}\r\n")
