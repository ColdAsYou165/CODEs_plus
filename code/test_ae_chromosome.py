'''
date:2022年11月24日
实验名称:
实验目的:测试train_ae_chromosome.py
描述:- [ ] 查看virtual_data本身的指标,比如统计父母类别,父母置信度,最高置信度种类,统计blendloss大小.
scale对结果的影响
'''
import os
import random
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpus", default="0")
parser.add_argument("--batch_size", type=int, default=128, help="苗师兄默认128")
parser.add_argument("--scale", type=int, default=4, help="crossover的比例为1/scale")
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
name_project = "test_ae_chromosome/v1"
args_str = get_args_str(args)
root_result, (root_pth, root_pic) = getResultDir(name_project=name_project,
                                                 name_args=args_str)
log = getLogger(formatter_str=None, root_filehandler=root_result + f"/logger.log")
log.info(str(args))
# writter = SummaryWriter(f"{root_result}/runs/run{datetime.now().strftime('%y-%m-%d,%H-%M-%S')}")
seed = random.randint(0, 2022)
setup_seed(seed)
log.info(f"seed={seed}")
# 数据集
batch_size = 8
# trainset_cifar10 = torchvision.datasets.CIFAR10(root="../data/cifar10", train=True, download=False,
#                                                 transform=transform_train_cifar_miao)
# testset_cifar10 = torchvision.datasets.CIFAR10(root="../data/cifar10", train=False, download=False,
#                                                transform=transform_test_cifar_miao)
# trainloader_cifar10 = DataLoader(trainset_cifar10, batch_size, shuffle=True, num_workers=2)
# testloader_cifar10 = DataLoader(testset_cifar10, batch_size, shuffle=True, num_workers=2)
num_classes = 10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 模型

model_g = AutoEncoder_Miao_crossover(num_classes).cuda()
model_g.apply(weights_init)
state_g = torch.load("../betterweights/ae_trainedbychromosome/ae_generatevirtual--scale2--wloss1e-3--epoch999.pth")
model_g.load_state_dict(state_g)

model_d = resnet_orig.ResNet18(num_classes=num_classes).cuda()
state_d = torch.load("../betterweights/resnet18_baseline_trainedbymiao_acc0.9532.pth")
model_d.load_state_dict(state_d)

# 优化器
criterion_blend = torch.nn.CrossEntropyLoss(reduce =False).cuda()
data = torch.load("../data/onebatch_cifar10.pt")
data, label = data["data"], data["label"]
print(data.shape, label.shape)

data1, data2, label1, label2, virtual_data1, virtual_data2, virtual_label = model_g.generate_virtual(data[:8],
                                                                                                     label[:8],
                                                                                                     scale=2,
                                                                                                     set_differentlabel=True,
                                                                                                     set_virtuallabel_uniform=False,
                                                                                                     set_test=True)
save_image(torch.concat([data1, data2, virtual_data1, virtual_data2], dim=0), root_pic + "/data.jpg")
print(data1.shape,data2.shape,virtual_data1.shape,virtual_data2.shape)
print(f"label1={label1}")
print(f"label2={label2}")
torch.set_printoptions(precision=2,sci_mode=False)
pred1 = model_d(virtual_data1).softmax(dim=1)
pred2 = model_d(virtual_data2).softmax(dim=1)
print(f"{pred1}")
loss=criterion_blend(pred1,virtual_label[:len(pred1)])
print(loss)