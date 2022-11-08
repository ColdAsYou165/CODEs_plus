'''
A能重建,B能重建,concat(A,B)能像两个类,遂为3crossloss
这个重建没用用mseloss,而是用更宽泛的crossloss,即不要像素级重建,而是要能识别出原来的类别就可以了.
'''
import os
import time
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument("--method", default="train_generate_virtual", help="执行哪个文件")
parser.add_argument("--method", default="train_3cross_wgan", help="执行哪个文件")  # train_3cross_wgan train_only_3cross
parser.add_argument("--gpus", default="0")
parser.add_argument("--epochs", type=int, default=1800)
parser.add_argument("--batch_size", type=int, default=128, help="苗师兄默认128")

parser.add_argument("--lr_g", type=float, default=6e-5, help="生成器的lr")  # 重构时候设置为0.0006,训练生成虚假图像的时候缩小10倍
parser.add_argument("--lr_dis", type=float, default=6e-5, help='wgan discrinator lr, default=0.0002')

parser.add_argument("--weight_loss_reconstruct", type=float, default=1, help="重构损失")
parser.add_argument("--weight_loss_w", type=float, default=1e-2, help="")

parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
# parser.add_argument("--seed", type=int, default=1)
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
from utils import *
from models import resnet_orig

# 起势
# v2调整 重构损失的权重
name_project = "train_ae_with3crossloss" + "/v3"
name_args = get_args_str(args)
root_result, (root_pth, root_pic) = getResultDir(name_project, name_args, "../results")
log = getLogger(None, root_result + f"/log.log")
log.info(f"v3建鹏师兄说取消预处理padding")
# setup_seed(args.seed)
# 数据集

num_classes = 10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
trainset_cifar10 = torchvision.datasets.CIFAR10(root="../data/cifar10", train=True, download=False,
                                                transform=transform_train_cifar_miao)
testset_cifar10 = torchvision.datasets.CIFAR10(root="../data/cifar10", train=False, download=False,
                                               transform=transform_test_cifar_miao)
trainloader_cifar10 = DataLoader(trainset_cifar10, args.batch_size, shuffle=True, num_workers=2)
testloader_cifar10 = DataLoader(testset_cifar10, args.batch_size, shuffle=True, num_workers=2)

origin_data, origin_label = next(iter(testloader_cifar10))
origin_data, origin_label = origin_data.cuda(), origin_label.cuda()
save_image(origin_data, root_pic + "/origin_data" + ".jpg")
img_black = torch.ones([1, 3, 32, 32]).cuda()
# 模型

model_g = AutoEncoder_Miao().cuda()
model_g.apply(weights_init)
state_g = torch.load("../betterweights/ae_miao_trainedbybclloss--epoch496--loss0.0006234363307940621.pth")
model_g.load_state_dict(state_g)
model_d = resnet_orig.ResNet18(num_classes=num_classes).cuda()
state = torch.load("../betterweights/resnet18_baseline_trainedbymiao_acc0.9532.pth")
model_d.load_state_dict(state)

discriminator = Discriminator_WGAN_miao_cifar10(set_sigmoid=False).cuda()  # set_sigmoid=False
discriminator.apply(weights_init)

# 优化器
criterion_cross = nn.CrossEntropyLoss().cuda()

optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=args.lr_dis, betas=(args.beta1, 0.999))
optimizer_g = torch.optim.Adam(model_g.parameters(), lr=args.lr_g, betas=(args.beta1, 0.999))

reconstruct = model_g(origin_data)
save_image(reconstruct, root_pic + f"/reconstrcu0.jpg")


# 这是不是生成了攻击样本?
def train_only_3cross():
    '''
    只用3crossloss,不用wloss
    '''
    for epoch in range(args.epochs):
        model_g.train()
        model_d.eval()
        loss_reconstruct_all = 0
        loss_virtual_all = 0
        for batch_idx, (data, label) in enumerate(trainloader_cifar10):
            data = data.cuda()
            label = label.cuda()
            reconstruct_data = model_g(data)
            pred_reconstruct = model_d(reconstruct_data)
            loss_reconstruct = criterion_cross(pred_reconstruct, label)  # 重构的还像原来的类别
            virtual_data, virtual_label = model_g.generate_virtual_v2(data, label, set_encoded_detach=True,
                                                                      train_generate=True, num_classes=num_classes)
            pred_virtual = model_d(virtual_data)
            loss_virtual = criterion_cross(pred_virtual, virtual_label)
            loss = loss_reconstruct * args.weight_loss_reconstruct + loss_virtual
            optimizer_g.zero_grad()
            loss.backward()
            optimizer_g.step()
            loss_reconstruct_all += loss_reconstruct.item()
            loss_virtual_all += loss_virtual.item()
        loss_reconstruct_all /= len(trainloader_cifar10)
        loss_virtual_all /= len(trainloader_cifar10)
        save_image(torch.concat([reconstruct_data, img_black, virtual_data], dim=0), root_pic + f"/recvir--{epoch}.jpg")
        log.info(f"[{epoch}/{args.epochs}]:loss_reconstruct={loss_reconstruct_all},loss_virtual={loss_virtual_all}")
        if batch_idx % 200 == 1:
            torch.save(model_g.state_dict(), root_pth + f"/model_g--epoch{epoch}--lossvir{loss_virtual_all:.2f}.pth")


def train_3cross_wgan():
    '''
    用3crossloss,用wloss
    '''
    mone = torch.tensor([-1.]).cuda()
    for epoch in range(args.epochs):
        model_g.train()
        model_d.eval()
        loss_reconstruct_all = 0
        loss_virtual_all = 0
        for batch_idx, (data, label) in enumerate(trainloader_cifar10):
            data = data.cuda()
            label = label.cuda()
            reconstruct_data = model_g(data)
            pred_reconstruct = model_d(reconstruct_data)
            loss_reconstruct = criterion_cross(pred_reconstruct, label)  # 重构的还像原来的类别
            virtual_data, virtual_label = model_g.generate_virtual_v2(data, label, set_encoded_detach=True,
                                                                      train_generate=True, num_classes=num_classes)
            # 鉴别器# real0 fake1
            pred_real = discriminator(data)
            optimizer_dis.zero_grad()
            pred_real.backward()
            pred_fake = discriminator(virtual_data.detach())
            pred_fake.backward(mone)
            optimizer_dis.step()

            # 生成器
            ## wloss
            optimizer_g.zero_grad()
            pred_wantreal = discriminator(virtual_data)
            (args.weight_loss_w * pred_wantreal).backward(retain_graph=True)
            ## crossloss
            pred_virtual = model_d(virtual_data)
            loss_virtual = criterion_cross(pred_virtual, virtual_label)
            loss = loss_reconstruct * args.weight_loss_reconstruct + loss_virtual
            loss.backward()
            optimizer_g.step()
            loss_reconstruct_all += loss_reconstruct.item()
            loss_virtual_all += loss_virtual.item()
        loss_reconstruct_all /= len(trainloader_cifar10)
        loss_virtual_all /= len(trainloader_cifar10)
        save_image(torch.concat([reconstruct_data, img_black, virtual_data], dim=0), root_pic + f"/recvir--{epoch}.jpg")
        log.info(f"[{epoch}/{args.epochs}]:loss_reconstruct={loss_reconstruct_all},loss_virtual={loss_virtual_all}")
        log.info(
            f"\treal,fake,wangreal:{pred_real.sigmoid().item(), pred_fake.sigmoid().item()},{pred_wantreal.sigmoid().item()}")
        if batch_idx % 200 == 1:
            torch.save(model_g.state_dict(), root_pth + f"/model_g--epoch{epoch}--lossvir{loss_virtual_all:.2f}.pth")


def train_only_crossloss():
    '''
    只用重建loss,重建loss为crossloss
    :return:
    '''
    for epoch in range(args.epochs):
        model_g.train()
        model_d.eval()
        loss_reconstruct_all = 0
        loss_virtual_all = 0
        for batch_idx, (data, label) in enumerate(trainloader_cifar10):
            data = data.cuda()
            label = label.cuda()
            reconstruct_data = model_g(data)
            pred_reconstruct = model_d(reconstruct_data)
            loss_reconstruct = criterion_cross(pred_reconstruct, label)  # 重构的还像原来的类别
            virtual_data, virtual_label = model_g.generate_virtual_v2(data, label, set_encoded_detach=True,
                                                                      train_generate=True, num_classes=num_classes)
            pred_virtual = model_d(virtual_data)
            loss_virtual = criterion_cross(pred_virtual, virtual_label)
            loss = loss_reconstruct * args.weight_loss_reconstruct + loss_virtual
            optimizer_g.zero_grad()
            loss.backward()
            optimizer_g.step()
            loss_reconstruct_all += loss_reconstruct.item()
            loss_virtual_all += loss_virtual.item()
        loss_reconstruct_all /= len(trainloader_cifar10)
        loss_virtual_all /= len(trainloader_cifar10)
        save_image(torch.concat([reconstruct_data, img_black, virtual_data], dim=0), root_pic + f"/recvir--{epoch}.jpg")
        log.info(f"[{epoch}/{args.epochs}]:loss_reconstruct={loss_reconstruct_all},loss_virtual={loss_virtual_all}")
        if batch_idx % 200 == 1:
            torch.save(model_g.state_dict(), root_pth + f"/model_g--epoch{epoch}--lossvir{loss_virtual_all:.2f}.pth")


if __name__ == "__main__":
    print(f"执行方法:{args.method}")
    if args.method == "train_only_3cross":
        train_only_3cross()
    elif args.method == "train_3cross_wgan":
        train_3cross_wgan()
    elif args.method == "train_only_crossloss":
        train_only_crossloss()
