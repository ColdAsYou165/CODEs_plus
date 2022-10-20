'''

'''
import os
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--method", default="train_generate_virtual", help="执行哪个文件")
parser.add_argument("--gpus", default="0")
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=128, help="苗师兄默认128")

parser.add_argument("--lr", type=float, default=6e-5, help="生成器的lr")  # 重构时候设置为0.0006,训练生成虚假图像的时候缩小10倍
parser.add_argument("--lr_dis", type=float, default=6e-5, help='wgan discrinator lr, default=0.0002')

parser.add_argument("--w_loss_weight", type=float, default=1, help="miaoshixiong 1e-5")
parser.add_argument("--blend_loss_weight", type=float, default=1., help="")
parser.add_argument("--chamfer_loss_weight", type=float, default=1., help="")

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
from utils import *
from models import resnet_orig

# 起势


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

model_g = AutoEncoder_Miao_containy().cuda()
model_g.apply(weights_init)

# 优化器
criterion_mse = torch.nn.MSELoss().cuda()
criterion_blend = torch.nn.CrossEntropyLoss().cuda()

origin_data, origin_label = next(iter(testloader_cifar10))
origin_data, origin_label = origin_data.cuda(), origin_label.cuda()
origin_label = F.one_hot(origin_label, num_classes).float()
# save_image(origin_data, results_pic_root + "/origin_data" + ".jpg")

optimizer_g = torch.optim.Adam(model_g.parameters(), lr=args.lr)


def train_reconstruct():
    # 训练正常的重构
    for epoch in range(args.epochs):
        model_g.train()
        loss_train, loss_test = 0, 0
        for batch_idx, (data, label) in enumerate(trainloader_cifar10):
            data = data.cuda()
            label = label.cuda()
            label = F.one_hot(label, num_classes).float().cuda()
            recontrust = model_g(data, label)
            loss = criterion_mse(recontrust, data)
            optimizer_g.zero_grad()
            loss.backward()
            optimizer_g.step()
            loss_train += loss.item()
        # scheduler.step()
        loss_train /= len(trainloader_cifar10)

        model_g.eval()
        for batch_idx, (data, label) in enumerate(testloader_cifar10):
            with torch.no_grad():
                data = data.cuda()
                label = label.cuda()
                label = F.one_hot(label, num_classes).float().cuda()
                recontrust = model_g(data, label)
                loss = criterion_mse(recontrust, data)
                loss_test += loss.item()
        loss_test /= len(testloader_cifar10)

        pic_reconstruct = model_g(origin_data, origin_label)
        save_image(pic_reconstruct, results_pic_root + f"/reconstruct--epoch{epoch}--loss{loss_train}.jpg")
        print(f"[{epoch}/{args.epochs}] : loss_train=", loss_train, "loss_test=", loss_test)
        if True and epoch > 180:
            torch.save(model_g.state_dict(),
                       results_pth_root + f"/ae_miao_trainedbybclloss--epoch{epoch}--loss{loss_test}.pth")


def test_reconstruct():
    #     拿能够重构的ae,检查一下,不同y会对应什么情况
    # 实验结果,不同y会产生一样的图像.也就是说,y被decoder忽视了.
    root_weight = "../betterweights/ae_miao_containy/ae_generatevirtual--epoch199--blend1.pth"
    model_g.load_state_dict(torch.load(root_weight))
    virtual_data = model_g(origin_data, origin_label)
    data, label = origin_data[0], origin_label[0]
    label = torch.argmax(label, dim=0)
    print(f"origin label is {label}")
    label = F.one_hot(torch.tensor([i for i in range(10)]), num_classes).cuda()
    data = data.expand(len(label), -1, -1, -1).cuda()
    reconstruct = model_g(data, label)
    img = torch.concat([reconstruct, virtual_data], dim=0)
    print(reconstruct.shape, img.shape)
    save_image(img, "./aecontainy.jpg")


def train_ae_bygan():
    '''
    像FaderNet那样引入一个鉴别器,使得encoder只包含内容信息
    :return:
    '''
    # 起势
    args_str = get_args_str(args)
    name_project = "train_ae_bygan_v1"

    writter = SummaryWriter(f"../runs/{name_project}/{datetime.now().strftime('%y-%m-%d,%H-%M-%S')}")
    writter.add_text("args", f"{name_project}\r\n{args}")

    root_result, (root_pth, root_pic) = getResultDir(name_project=name_project, name_args=args_str)

    log = getLogger(formatter_str=None, root_filehandler=root_result + f"/logger.log")
    log.info(str(args))
    # 模型
    model_d = Discriminator_FaderNet(in_channels=64, in_size=8).cuda()
    optimizer_d = torch.optim.Adam(model_d.parameters(), lr=args.lr_dis)
    # 小参数
    one = torch.tensor(1).float()
    mone = one * -1
    for epoch in range(args.epochs):
        loss_mse_all = 0
        loss_w_all = 0
        for batch_idx, (data, label) in enumerate(trainloader_cifar10):
            model_g.eval()
            model_d.train()
            data = data.cuda()
            label = label.cuda()
            # 训练鉴别器 real 0 fake 1
            optimizer_d.zero_grad()
            encoded = model_g.encoder(data)
            noise = torch.rand_like(encoded).cuda()
            pred_encoded = model_d(encoded.detach())
            pred_noise = model_d(noise)
            wloss_encoded = pred_encoded.mean()
            wloss_noise = pred_noise.mean()
            wloss_encoded.backward(one)
            wloss_noise.backward(mone)
            optimizer_d.step()

            # 训练生成器
            model_g.train()
            model_d.eval()
            ## 只保留类别信息
            pred_content = model_d(encoded)
            wloss = pred_content.mean()
            (args.w_loss_weight * wloss).backward(mone, retain_graph=True)
            loss_w_all += wloss.item()
            ## mseloss
            decoded = model_g.decoder(encoded, label)
            loss_mse = criterion_mse(decoded, data)
            loss_mse.backward()
            loss_mse_all += loss_mse.item()
            optimizer_g.step()
        loss_w_all /= len(trainloader_cifar10)
        loss_mse_all /= len(trainloader_cifar10)
        log.info(
            f"train[{epoch}/{args.epochs}]:wloss_noise={wloss_noise}wloss_encoded={wloss_encoded}\r\n\twloss={loss_w_all},mseloss={loss_mse_all}")
        if True and (epoch + 1) % 100 == 0:
            torch.save(model_g.state_dict(), root_pth + f"/model_ae--mse{loss_mse_all:.4f}--epoch{epoch}.pth")
        save_image(decoded, root_pic + f"/img--epoch{epoch}.jpg")


def train_generate_virtual():
    '''
    训练生成虚假图像
    :return:
    '''
    # 起势
    args_str = get_args_str(args)
    # v1只用wloss和blendloss
    # v2 只用wloss和blendloss,但是添加lr_g和lr_d相同或者为2倍以及,wloss:blendloss=1:1的约束,且epochs为1800
    # v2w2是观察一个epoch输出的图像长什么样子

    name_project = "ae_containy_generatevirtual_v2_w2"

    # log = getLogger(formatter_str=args_str, root_filehandler=f"../log/train_ae_containy_log.log")
    writter = SummaryWriter(f"../runs/{name_project}/{datetime.now().strftime('%y-%m-%d,%H-%M-%S')}")
    writter.add_text("args", f"{name_project}\r\n{args}")
    root_result, (root_pth, root_pic) = getResultDir(name_project=name_project, name_args=args_str)
    log = getLogger(formatter_str=None, root_filehandler=root_result + f"/logger.log")
    log.info(str(args))
    # 模型
    discriminator = Discriminator_WGAN_miao_cifar10(set_sigmoid=False).cuda()
    discriminator.apply(weights_init)

    model_d = resnet_orig.ResNet18(num_classes=10).cuda()
    state_d = torch.load("../betterweights/resnet18_baseline_trainedbymiao_acc0.9532.pth")
    model_d.load_state_dict(state_d)

    state_g = torch.load("../betterweights/ae_miao_containy.pth")
    model_g.load_state_dict(state_g)

    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=args.lr_dis, betas=(args.beta1, 0.999))
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
        loss_chamfer_all = 0
        for batch_idx, (data, label) in enumerate(trainloader_cifar10):
            data = data.cuda()
            label = label.cuda()
            # 训练鉴别器
            output_real = discriminator(data)
            optimizer_discriminator.zero_grad()
            output_real.backward(one)  # real 0
            pred_real_all += output_real.item()
            # 生成虚假图像
            virtual_data, virtual_label = model_g.generate_virtual(data, label, set_encode_detach=True,
                                                                   set_virtual_label_uniform=False)
            output_virtual = discriminator(virtual_data.detach())
            output_virtual.backward(mone)  # fake 1
            pred_virtual_all += output_virtual.item()
            optimizer_discriminator.step()

            # 训练生成器
            optimizer_g.zero_grad()
            ## wloss
            output_wantreal = discriminator(virtual_data)
            (args.w_loss_weight * output_wantreal).backward(one, retain_graph=True)  # 希望生成的virtual像真的,real为0
            loss_w_all += output_wantreal.item()

            ## blendwloss
            pred = model_d(virtual_data)
            loss_blend = criterion_blend(pred, virtual_label)
            # (args.blend_loss_weight * loss_blend).backward(retain_graph=True)
            (args.blend_loss_weight * loss_blend).backward()
            loss_blend_all += loss_blend.item()

            ## chamferloss

            optimizer_g.step()
        # 观察量
        pred_real_all /= len(trainloader_cifar10)
        pred_virtual_all /= len(trainloader_cifar10)
        loss_w_all /= len(trainloader_cifar10)
        loss_blend_all /= len(trainloader_cifar10)
        log.info(
            f"train[{epoch}/{args.epochs}] : pred_real={pred_real_all:.2f}, pred_virtual={pred_virtual_all:.2f}, "
            + f"loss_w={loss_w_all:.4f}, loss_blend={loss_blend_all:.4f}, loss_c={loss_chamfer_all}")

        save_image(virtual_data, root_pic + f"/virtualpic--epoch{epoch}.jpg")
        if True and (epoch + 1) % 200 == 0:
            torch.save(model_g.state_dict(), root_pth + f"/ae_generatevirtual--epoch{epoch}.pth")
        if epoch > 500:
            root_img_epoch = root_result + f"epochimg/epoch{epoch}"
            os.makedirs(root_img_epoch, exist_ok=True)
            with torch.no_grad():
                model_g.train()
                for batch_idx, (data, label) in enumerate(testloader_cifar10):
                    data = data.cuda()
                    label = label.cuda()
                    virtual_data, virtual_label = model_g.generate_virtual(data, label)
                    save_image(virtual_data, root_img_epoch + f"/testimg{batch_idx}.jpg")
                model_g.eval()
                for batch_idx, (data, label) in enumerate(trainloader_cifar10):
                    data = data.cuda()
                    label = label.cuda()
                    virtual_data, virtual_label = model_g.generate_virtual(data, label)
                    save_image(virtual_data, root_img_epoch + f"/trainimg{batch_idx}.jpg")


def test_generate_virtual():
    '''
    检查生成的虚假图像的质量
    :return:
    '''
    cifar100_train = torchvision.datasets.CIFAR100(root="../data/cifar100", train=True, download=False,
                                                   transform=transform_train_cifar_miao)
    cifar100_test = torchvision.datasets.CIFAR100(root="../data/cifar100", train=False, download=False,
                                                  transform=transform_test_cifar_miao)
    svhn_train = torchvision.datasets.SVHN(root="../data/svhn", split="train", download=False,
                                           transform=transform_train_cifar_miao)
    svhn_test = torchvision.datasets.SVHN(root="../data/svhn", split="test", download=False,
                                          transform=transform_test_cifar_miao)
    trainloader_cifar100 = torch.utils.data.DataLoader(cifar100_train, batch_size=args.batch_size, shuffle=True,
                                                       num_workers=2)
    testloader_cifar100 = torch.utils.data.DataLoader(cifar100_test, batch_size=args.batch_size, shuffle=True,
                                                      num_workers=2)
    trainloader_svhn = torch.utils.data.DataLoader(svhn_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testloader_svhn = torch.utils.data.DataLoader(svhn_test, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # batch_size128--beta10.5--blend_loss_weight1.0--epochs1800--gpus'0'--lr6e-05--lr_dis6e-05--method'train_generate_virtual'--w_loss_weight1
    # root_pth = "/mnt/data/maxiaolong/CODEsSp/results/ae_containy_generatevirtual_v2_w1/argsbatch_size128--beta10.5--blend_loss_weight1.0--epochs1800--gpus'0'--lr6e-05--lr_dis6e-05--method'train_generate_virtual'--w_loss_weight1/pthae_generatevirtual--epoch1599.pth"
    root_pth = "/mnt/data/maxiaolong/CODEsSp/results/ae_containy_generatevirtual_v2_w1/argsbatch_size128--beta10.5--blend_loss_weight0.001--epochs1800--gpus'1'--lr6e-05--lr_dis6e-05--method'train_generate_virtual'--w_loss_weight1/pthae_generatevirtual--epoch1799.pth"
    model_g.load_state_dict(torch.load(root_pth))
    torch.no_grad()
    # model_g.eval()
    root_result = f"../results/test_generate_virtual"
    root_img = root_result + "/img"
    os.makedirs(root_img, exist_ok=True)
    os.makedirs(root_result, exist_ok=True)
    # model_d=
    for batch_idx, (data, label) in enumerate(testloader_cifar10):
        data = data.cuda()
        label = label.cuda()
        virtual_data, virtual_label = model_g.generate_virtual(data, label)
        save_image(virtual_data, root_img + f"/testimg{batch_idx}.jpg")
        print(virtual_data.shape)
    for batch_idx, (data, label) in enumerate(trainloader_cifar10):
        data = data.cuda()
        label = label.cuda()
        virtual_data, virtual_label = model_g.generate_virtual(data, label)
        print(virtual_data.shape)
        save_image(virtual_data, root_img + f"/trainimg{batch_idx}.jpg")


def supress_by_virtual():
    '''
    压制试验
    :return:
    '''
    root_pth = "/mnt/data/maxiaolong/CODEsSp/results/ae_containy_generatevirtual_v2_w1/argsbatch_size128--beta10.5--blend_loss_weight1.0--epochs1800--gpus'0'--lr6e-05--lr_dis6e-05--method'train_generate_virtual'--w_loss_weight1/pthae_generatevirtual--epoch1799.pth"
    state_g = torch.load(root_pth)
    model_g.load_state_dict(state_g)
    model_g.eval()
    for epoch in range(args.epochs):
        for batch_idx, (data, label) in enumerate(trainloader_cifar10):
            pass


if __name__ == "__main__":
    # args.method = "train_generate_virtual"
    print("执行方法:", args.method)
    if args.method == "train_generate_virtual":
        train_generate_virtual()
    elif args.method == "train_reconstruct":
        train_reconstruct()
    elif args.method == "test_reconstruct":
        test_reconstruct()
    elif args.method == "train_ae_bygan":
        train_ae_bygan()
    elif args.method == "test_generate_virtual":
        test_generate_virtual()
    else:
        exit(-1)
