'''
date:2022年12月15日
实验名称:
实验目的:模仿染色体交叉环节生成虚假样本
描述:label按照权重
'''
import os
import random
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpus", default="0")
parser.add_argument("--epochs", type=int, default=1800)
parser.add_argument("--batch_size", type=int, default=128, help="苗师兄默认128")

parser.add_argument("--lr", type=float, default=6e-5, help="生成器的lr")  # 重构时候设置为0.0006,训练生成虚假图像的时候缩小10倍
parser.add_argument("--lr_dis", type=float, default=6e-5, help='对抗分类器的 lr, default=0.0002')  # virtual时候6e-5

parser.add_argument("--w_loss_weight", type=float, default=1, help="miaoshixiong 1e-5")  # 训练aegenerate virtual时用的
parser.add_argument("--blend_loss_weight", type=float, default=1., help="")
parser.add_argument("--scale", type=int, default=8, help="crossover的比例为1/scale")
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm
# import torch.backends.cudnn as cudnn
import sys
from datetime import datetime
from model import *
from models.ae_miao_chromosome import *
from utils import *
from models import resnet_orig

# 起势
# 不带v的为实验版本
# 目前吧scale改成2 blendloss为1生成的还是像正常样本,所以调小wloss的权重
# v3 鉴别器优化器从Adam换成RMSprop
# v481 换成可学习参数
name_project = "train_ae_chromosome/xiajibaxie"
args_str = get_args_str(args)
root_result, (root_pth, root_pic) = getResultDir(name_project=name_project,
                                                 name_args=args_str)
log = getLogger(formatter_str=None, root_filehandler=root_result + f"/logger.log")
log.info("virtual target 按照权重")
log.info(str(args))
# writter = SummaryWriter(f"{root_result}/runs/run{datetime.now().strftime('%y-%m-%d,%H-%M-%S')}")
seed = random.randint(0, 2022)
setup_seed(seed)
log.info(f"seed={seed}")

# 数据集

num_classes = 10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
trainset_cifar10 = torchvision.datasets.CIFAR10(root="../data/cifar10", train=True, download=False,
                                                transform=transform_train_cifar_miao)
testset_cifar10 = torchvision.datasets.CIFAR10(root="../data/cifar10", train=False, download=False,
                                               transform=transform_test_cifar_miao)
trainloader_cifar10 = DataLoader(trainset_cifar10, args.batch_size, shuffle=True, num_workers=2)
testloader_cifar10 = DataLoader(testset_cifar10, args.batch_size, shuffle=True, num_workers=2)

# 模型

model_g = AutoEncoder_Miao_crossover(num_classes).cuda()
model_g.apply(weights_init)
state_g = torch.load("../betterweights/ae_miao_mseloss.pth")
model_g.load_state_dict(state_g)

discriminator = DCGAN_D(isize=32, nz=100, nc=3, ndf=64, ngpu=1).cuda()
discriminator.apply(weights_init)

model_d = resnet_orig.ResNet18(num_classes=num_classes).cuda()
state_d = torch.load("../betterweights/resnet18_baseline_trainedbymiao_acc0.9532.pth")
model_d.load_state_dict(state_d)

# 优化器
criterion_blend = torch.nn.CrossEntropyLoss().cuda()
optimizer_g = torch.optim.Adam(model_g.parameters(), lr=args.lr)
# optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=args.lr_dis, betas=(args.beta1, 0.999))
optimizer_discriminator = torch.optim.RMSprop(discriminator.parameters(), lr=args.lr_dis)

origin_data, origin_label = next(iter(trainloader_cifar10))
origin_data, origin_label = origin_data.cuda(), origin_label.cuda()
# torch.save({"data": origin_data, "label": origin_label}, "../data/onebatch_cifar10.pt")
# print(f"保存成功")
origin_label = F.one_hot(origin_label, num_classes).float()
origin_reconstruct = model_g(origin_data)
save_image(torch.concat([origin_data, origin_reconstruct], dim=0), root_pic + "/origin.jpg")

# 小参数
one = torch.FloatTensor([1])
mone = one * -1
one, mone = one.cuda(), mone.cuda()
model_d.eval()
# real 0 fake 1
for epoch in range(args.epochs):
    model_g.train()
    # 观察量
    pred_real_all, pred_virtual_all = 0, 0
    loss_w_all = 0
    loss_blend_all = 0
    # loss_chamfer_all = 0

    for batch_idx, (data, label) in enumerate(trainloader_cifar10):
        data = data.cuda()
        label = label.cuda()
        # 训练鉴别器
        output_real = discriminator(data)
        optimizer_discriminator.zero_grad()
        # output_real.backward(one)  # real 0
        pred_real_all += output_real.item()
        # 生成虚假图像
        virtual_data, virtual_label = model_g.generate_virtual_anquanzhong(data, label, set_differentlabel=True,
                                                                           set_virtuallabel_uniform=False, scale=args.scale)
        output_virtual = discriminator(virtual_data.detach())
        (output_real - output_virtual).backward()
        pred_virtual_all += output_virtual.item()
        optimizer_discriminator.step()

        # 训练生成器
        optimizer_g.zero_grad()
        ## wloss
        output_wantreal = discriminator(virtual_data)
        # (args.w_loss_weight * output_wantreal).backward(retain_graph=True)  # 希望生成的virtual像真的,real为0
        loss_w_all += output_wantreal.item()

        ## blendwloss
        pred = model_d(virtual_data)
        loss_blend = criterion_blend(pred, virtual_label)
        # (args.blend_loss_weight * loss_blend).backward(retain_graph=True)
        (args.w_loss_weight * output_wantreal + args.blend_loss_weight * loss_blend).backward()
        loss_blend_all += loss_blend.item()

        optimizer_g.step()
    # 观察量
    pred_real_all /= len(trainloader_cifar10)
    pred_virtual_all /= len(trainloader_cifar10)
    loss_w_all /= len(trainloader_cifar10)
    loss_blend_all /= len(trainloader_cifar10)
    log.info(
        f"train[{epoch}/{args.epochs}] : pred_real={pred_real_all:.2f}, pred_virtual={pred_virtual_all:.2f}, "
        + f"loss_w={loss_w_all:.4f}, loss_blend={loss_blend_all:.4f}")
    save_image(torch.concat([data, virtual_data], dim=0),
               root_pic + f"/ae_chromosome_trainedtogeneratevirtual--epoch{epoch}.jpg")
    if True and (epoch + 1) % 200 == 0:
        torch.save(model_g.state_dict(), root_pth + f"/ae_generatevirtual--epoch{epoch}.pth")
