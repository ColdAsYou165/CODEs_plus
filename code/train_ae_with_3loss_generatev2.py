''''
v2按照老师要求写的 加法
去掉chamfer loss
ae使用lr0.0006训练的
'''
import os
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lr_g", type=float, default=0.00006, help="model_g的lr")
parser.add_argument('--lr_dis', type=float, default=0.00006, help='wgan discrinator lr, default=0.0002')
parser.add_argument('--lr_scale', type=float, default=1e4, help='wgan discrinator lr, default=0.0002')
parser.add_argument("--optimizer", default="Adam", help="Adam SGD")
parser.add_argument("--epochs", type=int, default=1800)
parser.add_argument("--gpus", default="0")
parser.add_argument("--batch_size", type=int, default=128, help="苗师兄默认128")
# wgan的 discriminator
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--w_loss_weight', type=float, default=1e-7, help='wloss上加的权重,苗师兄wgan是1e-5')
parser.add_argument('--blend_loss_weight', type=float, default=1, help='cross_loss上加的权重')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

if args.lr_dis == 0 or args.lr_dis < 0:
    args.lr_dis = args.lr_g / args.lr_scale
print(str(args))

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
import chamfer3D.dist_chamfer_3D
from tqdm import tqdm
# import torch.backends.cudnn as cudnn
import sys
from model import *
from utils import *
from models import resnet_orig

# 起势
name_args = get_args_str(args)

# train_ae_with_3loss_generatev2——v1 两个latent vector相加'
# train_ae_with_3loss_generatev2_v2去掉chamfer loss
# v3 还是只使用wloss和blendloss,但是wloss权重要调小
# v4只有blendloss
name_project = f"train_ae_with_3loss_generatev2_v4"
results_root = f"../results/{name_project}/{name_args}"
os.makedirs(results_root, exist_ok=True)
file = open(results_root + "/args.txt", "w")
file.write(f"{args}")
file.close()
results_pic_root = results_root + "/pic"
results_pth_root = results_root + "/pth"
os.makedirs(results_pic_root, exist_ok=True)
os.makedirs(results_pth_root, exist_ok=True)
writer = SummaryWriter()
writer.add_text("实验描述", f"{name_project},{args}")

# 数据集

num_classes = 10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
trainset = torchvision.datasets.CIFAR10(root="../data/cifar10", train=True, download=False,
                                        transform=transform_train_cifar_miao)
testset = torchvision.datasets.CIFAR10(root="../data/cifar10", train=False, download=False,
                                       transform=transform_test_cifar_miao)
trainloader = DataLoader(trainset, args.batch_size, shuffle=True, num_workers=2)
testloader = DataLoader(testset, args.batch_size, shuffle=True, num_workers=2)

# 模型
## wgan的dis是否需要sigmoid,不能要sigmoid!
discriminator = Discriminator_WGAN_miao_cifar10(set_sigmoid=False).cuda()  # set_sigmoid=False
discriminator.apply(weights_init)

model_g = AutoEncoder_Miao().cuda()

model_g.apply(weights_init)
root_ae_1 = "/mnt/data/maxiaolong/CODEsSp/results/train_ae_with_mseloss_v3/args--lr0.0006/pth/ae_miao_trainedbybclloss--epoch499--loss0.000613753743546343.pth"
state_g = torch.load(root_ae_1)
model_g.load_state_dict(state_g)

model_d = resnet_orig.ResNet18(num_classes=10).cuda()
state = torch.load("../betterweights/resnet18_baseline_trainedbymiao_acc0.9532.pth")
model_d.load_state_dict(state)

# 优化器
criterion_blend = nn.CrossEntropyLoss().cuda()
# chamLoss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()
### 师兄的鉴别器和生成器 的优化器也都是用的Adam
optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=args.lr_dis, betas=(args.beta1, 0.999))
optimizer_g = torch.optim.Adam(model_g.parameters(), lr=args.lr_g, betas=(args.beta1, 0.999))

origin_data, origin_label = next(iter(testloader))
origin_data, origin_label = origin_data.cuda(), origin_label.cuda()
save_image(origin_data, results_pic_root + "/origin_data" + ".jpg")


# 训练
def ae(epoch):
    one = torch.FloatTensor([1])
    mone = one * -1
    one, mone = one.cuda(), mone.cuda()
    pred_dis_real_all = 0
    pred_dis_fake_all = 0
    loss_chamfer_all = 0
    loss_cross_all = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # 更新鉴别器
        inputs = inputs.cuda()
        targets = targets.cuda()
        discriminator.zero_grad()
        real_cpu = inputs.cuda()
        output = discriminator(real_cpu)
        d_loss_real = output
        d_loss_real.backward(one)  # real为0
        pred_dis_real = output.sigmoid()  # 观察量 dis对real的pred,问题是苗师兄里面接了mean
        pred_dis_real_all += pred_dis_real.item()
        # ##encoder之后先detach再decoder得到虚假图像
        decoded, virtual_label = model_g.generate_virtual_v2(inputs, targets, set_encoded_detach=True,
                                                             train_generate=True, num_classes=num_classes)
        output = discriminator(decoded.detach())  # 注意detach
        d_loss_fake = output
        d_loss_fake.backward(mone)  # fake为1
        pred_dis_fake = output.sigmoid()  # 观察量 dis对real的pred,问题是苗师兄里面接了mean
        pred_dis_fake_all += pred_dis_fake.item()
        # Update D
        optimizer_dis.step()

        # 更新生成器
        model_g.zero_grad()

        output = discriminator(decoded)
        g_loss = args.w_loss_weight * output
        g_loss.backward(one, retain_graph=True)  # 怎么增加类别信息嫩,gan语义生成错在哪里嫩??

        ##计算crossentropyloss
        pred_model_d = model_d(decoded)

        loss_cross = criterion_blend(pred_model_d, virtual_label)
        loss_cross_all += loss_cross.item()
        (loss_cross * args.blend_loss_weight).backward(retain_graph=True)

        optimizer_g.step()
    # 观察量
    pred_dis_real_all /= len(trainloader)
    pred_dis_fake_all /= len(trainloader)
    loss_chamfer_all /= len(trainloader)
    loss_cross_all /= len(trainloader)
    print(
        f"[{epoch}/{args.epochs}]:loss_chamfer_all={loss_chamfer_all:3f},pred_dis_real_all={pred_dis_real_all:.3f},pred_dis_fake_all={pred_dis_fake_all:.3f},loss_cross_all={loss_cross_all:.3f}")
    writer.add_scalar("chamfer_loss", loss_chamfer_all, epoch)
    writer.add_scalar("loss_cross", loss_cross_all, epoch)
    # 每个epoch生成并保存一张虚假图片
    virtual_data, virtual_label = model_g.generate_virtual_v2(origin_data, origin_label, set_encoded_detach=True,
                                                              train_generate=True, num_classes=num_classes)
    save_image(virtual_data, results_pic_root + f"/virpic_--epoch{epoch}--chamferloss{loss_chamfer_all:.3f}.jpg")

    # 保存模型权重
    if True and (epoch + 1) % 100 == 0:
        state = {"model": model_g.state_dict(), "loss": loss_chamfer_all}
        torch.save(state, results_pth_root + f"/model_chamfer_and_wloss--epoch{epoch}.pth")


# 训练
def ae_blend(epoch):
    one = torch.FloatTensor([1])
    mone = one * -1
    one, mone = one.cuda(), mone.cuda()
    pred_dis_real_all = 0
    pred_dis_fake_all = 0
    loss_chamfer_all = 0
    loss_cross_all = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # 更新鉴别器
        inputs = inputs.cuda()
        targets = targets.cuda()
        # ##encoder之后先detach再decoder得到虚假图像
        decoded, virtual_label = model_g.generate_virtual_v2(inputs, targets, set_encoded_detach=True,
                                                             train_generate=True, num_classes=num_classes)

        ##计算crossentropyloss
        pred_model_d = model_d(decoded)

        loss_cross = criterion_blend(pred_model_d, virtual_label)
        loss_cross_all += loss_cross.item()
        (loss_cross * args.blend_loss_weight).backward()

        optimizer_g.step()
    # 观察量
    pred_dis_real_all /= len(trainloader)
    pred_dis_fake_all /= len(trainloader)
    loss_chamfer_all /= len(trainloader)
    loss_cross_all /= len(trainloader)
    print(
        f"[{epoch}/{args.epochs}]:loss_chamfer_all={loss_chamfer_all:3f},pred_dis_real_all={pred_dis_real_all:.3f},pred_dis_fake_all={pred_dis_fake_all:.3f},loss_cross_all={loss_cross_all:.3f}")
    writer.add_scalar("chamfer_loss", loss_chamfer_all, epoch)
    writer.add_scalar("loss_cross", loss_cross_all, epoch)
    # 每个epoch生成并保存一张虚假图片
    virtual_data, virtual_label = model_g.generate_virtual_v2(origin_data, origin_label, set_encoded_detach=True,
                                                              train_generate=True, num_classes=num_classes)
    save_image(virtual_data, results_pic_root + f"/virpic_--epoch{epoch}--chamferloss{loss_chamfer_all:.3f}.jpg")

    # 保存模型权重
    if True and (epoch + 1) % 100 == 0:
        state = {"model": model_g.state_dict(), "loss": loss_chamfer_all}
        torch.save(state, results_pth_root + f"/model_chamfer_and_wloss--epoch{epoch}.pth")


for epoch in range(args.epochs):
    ae_blend(epoch)
