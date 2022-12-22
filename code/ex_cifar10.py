import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
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

num_classes = 10
batch_size = 8
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
model_g = AutoEncoder_Miao_crossover(num_classes).cuda()
# state_g = "../betterweights/ae_trainedbychromosome/ae_generatevirtual--scale2--wloss1e-3--epoch1799.pth"
state_g = "../results/train_ae_chromosome_blend2tangloss/v1/argsbatch_size128--beta10.5--epochs1800--gpus'4'--lr6e-05--lr_dis6e-05--scale8--tang_loss_weight1.0--w_loss_weight1/pth/ae_generatevirtual--epoch1799.pth"
model_g.load_state_dict(torch.load(state_g))

model_d = resnet_orig.ResNet18(num_classes=num_classes).cuda()
state_d = torch.load("../betterweights/resnet18_baseline_trainedbymiao_acc0.9532.pth")
model_d.load_state_dict(state_d)
torch.set_printoptions(precision=2, sci_mode=False, linewidth=120)
np.set_printoptions(precision=2, suppress=True, linewidth=120)

cifar10_train = torchvision.datasets.CIFAR10(root="../data/cifar10", train=True, download=False,
                                             transform=transform_train_cifar_miao)
cifar10_test = torchvision.datasets.CIFAR10(root="../data/cifar10", train=False, download=False,
                                            transform=transform_test_cifar_miao)
trainloader_cifar10 = torch.utils.data.DataLoader(cifar10_train, batch_size=batch_size, shuffle=True,
                                                  num_workers=2)
testloader_cifar10 = torch.utils.data.DataLoader(cifar10_test, batch_size=batch_size, shuffle=True, num_workers=2)
root_img = "../results/debugcifar10"
os.makedirs(root_img, exist_ok=True)
'''
for batch_idx, (data, label) in enumerate(trainloader_cifar10):
    data = data.cuda()
    label = label.cuda()
    all_data_all = []
    for i in range(len(data) - 1):
        print(f"标号{i}")
        virtual_data0, virtual_data1 = model_g.generate_one_virtual(data[i], data[i + 1], scale=8)

        all_data = torch.stack([data[i], data[i + 1], virtual_data0.squeeze(), virtual_data1.squeeze()])
        print(all_data.shape)
        pred = model_d(all_data).softmax(1)
        print(
            f"类别(依次为父,母,由父生成的virtual图像,由母生成的virtual图像):{classes[label[i].item()], label[i].item()},{classes[label[i + 1].item()], label[i + 1].item()},{classes[torch.topk(pred[-2], k=1)[1].int()], torch.topk(pred[-2], k=1)[1].item()},{classes[torch.topk(pred[-1], k=1)[1].int()], torch.topk(pred[-1], k=1)[1].item()}")
        print(
            f"虚假图像在分类器中的top2置信度:{torch.topk(pred[-2], k=2)[0].detach().cpu().numpy(), torch.topk(pred[-1], k=2)[0].detach().cpu().numpy()}")
        print("置信度(依次为父,母,由父生成的virtual图像,由母生成的virtual图像):\n", pred.cpu().detach().numpy())
        print(f"图像(依次为父,母,由父生成的virtual图像,由母生成的virtual图像):\n")
        imshow(all_data)
        # save_image(all_data, root_img + f"/img{batch_idx}-{i}.jpg")
        all_data_all += all_data
        # imshow(all_data)
        # print("-" * 60)
    all_data_all = torch.concat(all_data_all, dim=0)

    break
'''
file = torch.load("../data/onebatch_cifar10.pt")
data, label = file["data"], file["label"]
data = data.cuda()
label = label.cuda()
for i in range(len(data) - 1):
    print(f"标号{i}")
    virtual_data0, virtual_data1 = model_g.generate_one_virtual(data[i], data[i + 1], scale=8)

    all_data = torch.stack([data[i], data[i + 1], virtual_data0.squeeze(), virtual_data1.squeeze()])
    pred = model_d(all_data).softmax(1)
    print(
        f"类别(依次为父,母,由父生成的virtual图像,由母生成的virtual图像):{classes[label[i].item()], label[i].item()},{classes[label[i + 1].item()], label[i + 1].item()},{classes[torch.topk(pred[-2], k=1)[1].int()], torch.topk(pred[-2], k=1)[1].item()},{classes[torch.topk(pred[-1], k=1)[1].int()], torch.topk(pred[-1], k=1)[1].item()}")
    print(
        f"虚假图像在分类器中的top2置信度:{torch.topk(pred[-2], k=2)[0].detach().cpu().numpy(), torch.topk(pred[-1], k=2)[0].detach().cpu().numpy()}")
    print("置信度(依次为父,母,由父生成的virtual图像,由母生成的virtual图像):\n", pred.cpu().detach().numpy())
    print(f"图像(依次为父,母,由父生成的virtual图像,由母生成的virtual图像):\n")
    imshow(all_data)
    print("-" * 60)