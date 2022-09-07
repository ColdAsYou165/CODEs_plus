'''
使用CIFAR10数据集训练一个正常的ResNet
keras指出ResNet32 200轮	acc=92.46 %	论文官方acc=92.49 %
数据集描述:
cifar10 有10类,图像大小为32*32
**实际训练效果**训练的效果不错,resnet18 epoch200时候test acc=0.956,loss=0.1652
https://github.com/kuangliu/pytorch-cifar
'''
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import sys
from model import ResNet18, ResNet34
from utils import *
from models import resnet_orig

name_project = "train_resnet_github"
root_result = f"../results/{name_project}/{time.time():.2f}"
os.makedirs(root_result)
root_pth = root_result + "/pth"
os.makedirs(root_pth)
root_runs = root_result + f"/runs"
writer = SummaryWriter(root_runs)
writer.add_text("实验描述", "训练resnet18,transform 只有totensor,一定要训练出来啊")
# 超参数
batch_size = 128
epochs = 200
lr = 0.1
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
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
model = resnet_orig.ResNet18(num_classes=10).cuda()
# model = torch.nn.DataParallel(model)

cudnn.benchmark = True
criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)  # 5的10的-4次方,0.0005
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


def train_and_test():
    # 训练 测试
    for epoch in range(epochs):
        # 训练
        model.train()
        loss_train = 0
        acc_train = 0
        for batch, (data, label) in enumerate(tqdm(trainloader_cifar10)):
            data = data.cuda()
            label = label.cuda()
            optimizer.zero_grad()
            outputs = model(data)
            outputs_label = torch.argmax(outputs, dim=1)
            acc_train += (outputs_label == label).int().sum().item()
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        loss_train = loss_train / len(trainloader_cifar10)
        acc_train = acc_train / len(cifar100_train)
        print(f"train{epoch},acc={acc_train},loss={loss_train}")
        # 测试
        model.eval()
        acc_test = 0
        loss_test = 0
        with torch.no_grad():
            for batch, (data, label) in enumerate(tqdm(testloader_cifar10)):
                data = data.cuda()
                label = label.cuda()
                outputs = model(data)
                outputs_label = torch.argmax(outputs, dim=1)
                loss = criterion(outputs, label)
                acc_test += (outputs_label == label).int().sum().item()
                loss_test += loss.item()
        acc_test /= (len(cifar10_test))
        loss_test /= len(testloader_cifar10)
        print(f"test epoch{epoch},acc={acc_test},loss={loss_test}")
        writer.add_scalars("loss", {"loss_train": loss_train, "loss_test": loss_test}, epoch)
        writer.add_scalars("acc", {"acc_train": acc_train, "acc_test": acc_test}, epoch)
        # 保存
        torch.save(model.state_dict(), root_pth + f"/resnet_github--acc{acc_test:.4f}.pth")
        scheduler.step()


# train_and_test()

def test(epoch=1):
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                            download=False, transform=transform_tensor_norm)
    testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                           download=False, transform=transform_tensor_norm)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=512,
                                              shuffle=True, num_workers=0)

    testloader = torch.utils.data.DataLoader(testset, batch_size=512,
                                             shuffle=True, num_workers=0)
    model.load_state_dict(torch.load("../betterweights/mymodel__epoch30__transform_tensor_norm.pth")["model"])
    model.eval()
    acc_test = 0
    loss_test = 0
    with torch.no_grad():
        for batch, (data, label) in enumerate(tqdm(testloader)):
            data = data.cuda()
            label = label.cuda()
            outputs = model(data)
            outputs_label = torch.argmax(outputs, dim=1)
            loss = criterion(outputs, label)
            acc_test += (outputs_label == label).int().sum().item()
            loss_test += loss.item()
    acc_test /= (len(testset))
    loss_test /= len(testloader)
    print(f"test epoch{epoch},acc={acc_test},loss={loss_test}")


# test()
if __name__ == "__main__":
    train_and_test()
    # test()
