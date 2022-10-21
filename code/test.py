'''
测试单独使用一个文件
'''
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpus", default="0")
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


def test_ae_reconstruct(model=AutoEncoder_Miao().cuda(), trainloader=trainloader_cifar10,
                        testloader=testloader_cifar10, root=""):
    '''
    检查ae重构的mseloss
    :param model:
    :param trainloader:
    :param testloader:
    :param root:
    :return:
    '''
    # "../betterweights/ae_miao_reconstruct_v0--mseloss0.000919.pth"
    # "../betterweights/ae_miao_trainedbybclloss--epoch496--loss0.0006234363307940621.pth"现在用这个2022年9月9日
    model_state = torch.load(root)
    model.load_state_dict(model_state)

    loss_mse_all = 0
    for batch_idx, (data, label) in enumerate(tqdm(trainloader)):
        data = data.cuda()
        label = label.cuda()
        reconstruct = model(data)
        loss_mse = criterion_mse(data, reconstruct)
        loss_mse_all += loss_mse.item()
    loss_mse_all /= len(trainloader)
    print("loss_mse_train=", loss_mse_all)

    loss_mse_all = 0
    for batch_idx, (data, label) in enumerate(tqdm(testloader)):
        data = data.cuda()
        label = label.cuda()
        reconstruct = model(data)
        loss_mse = criterion_mse(data, reconstruct)
        loss_mse_all += loss_mse.item()
    loss_mse_all /= len(testloader)
    print("loss_mse_test=", loss_mse_all)


def view_virtual(model=AutoEncoder_Miao().cuda(), root=""):
    state_g = torch.load(
        f"/mnt/data/maxiaolong/CODEsSp/results/train_ae_with_3loss_generatev3_v1/argslr_g6e-05--lr_dis6e-05--lr_scale10000.0--optimizer'Adam'--epochs1800--gpus'7'--batch_size128--beta10.5--w_loss_weight1e-05--blend_loss_weight1e-06/pth/model_chamfer_and_wloss--epoch199.pth")[
        "model"]
    os.makedirs("./pic", exist_ok=True)
    model.load_state_dict(state_g)
    for data, label in trainloader_cifar10:
        data = data.cuda()
        label = label.cuda()
        virtual_data = model(data)
        save_image(virtual_data, f"./pic/virtual_data,jpg")


def test_train_ae_forkfadernet_differnety():
    '''
    测试相同图像不同y会生成不一样的图像吗?
    :return:
    '''
    model_g = AutoEncoder_Miao_containy(num_classes=10).cuda()
    root = "../results/train_ae_forkfadernet_v1/"
    files = os.listdir(root)
    data, label = next(iter(trainloader_cifar10))
    data = data.cuda()
    label = label.cuda()
    label = torch.tensor([i for i in range(10)]).long().cuda()
    for file in files:
        root_pth = root + file + "/pth" + "/ae_ae_forkFaderNet--epoch1199.pth"
        root_result = f"../results/test_train_ae_forkfadernet_differnety/{file}/"
        os.makedirs(root_result, exist_ok=True)
        if not os.path.exists(root_pth):
            continue
        model_g.load_state_dict(torch.load(root_pth))
        for i in range(10):
            img = data[i].expand([len(label), -1, -1, -1])
            # virtual_data, virtual_label = model_g.generate_virtual(img, label)
            virtual_data = model_g(img, label)
            print(virtual_data.shape)
            save_image(virtual_data, root_result + f"virtual_data{i}.jpg")


if __name__ == "__main__":
    pass
    # test_resnet_acc_and_mmc(model=model_resnet18, root="../betterweights/resnet18_baseline_trainedbymiao_acc0.9532.pth")
    # test_ae_reconstruct(root="../betterweights/ae_miao_trainedbybclloss--epoch496--loss0.0006234363307940621.pth")

    # view_virtual()
    test_train_ae_forkfadernet_differnety()
    '''root = "../results/train_ae_forkfadernet_v1/"
    files = os.listdir(root)
    for file in files:
        # pth=os.listdir(root + file)[2]
        pth = "pth"
        a = os.listdir(root + file + "/" + pth)
        # print(root + file + "/" + pth)
        print(a)'''
