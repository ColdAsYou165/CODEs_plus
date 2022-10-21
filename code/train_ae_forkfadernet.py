'''
仿照Fadernet的结构,鉴别器使用的是分类器
'''
import os
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--method", default="train_reconstruct", help="执行哪个任务")
parser.add_argument("--gpus", default="0")
parser.add_argument("--epochs", type=int, default=1200)
parser.add_argument("--batch_size", type=int, default=128, help="苗师兄默认128")

parser.add_argument("--lr", type=float, default=6e-5, help="生成器的lr")  # 重构时候设置为0.0006,训练生成虚假图像的时候缩小10倍
parser.add_argument("--lr_dis", type=float, default=6e-5, help='Fadernet为0.0002')

# parser.add_argument("--w_loss_weight", type=float, default=1, help="miaoshixiong 1e-5")
parser.add_argument("--cross_loss_weight", type=float, default=1., help="")
# parser.add_argument("--chamfer_loss_weight", type=float, default=1., help="")
parser.add_argument("--method_label", type=int, default=0, help="0的话就为FaderNet中的随机label")
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
from torch.autograd import Variable
from tqdm import tqdm
# import torch.backends.cudnn as cudnn
import sys
from datetime import datetime
from model import *
from utils import *
from models import resnet_orig

# 起势
args_str = get_args_str(args)
name_project = "train_ae_forkfadernet_v1"

writter = SummaryWriter(f"../runs/{name_project}/{datetime.now().strftime('%y-%m-%d,%H-%M-%S')}")
writter.add_text("args", f"{name_project}\r\n{args}")

root_result, (root_pth, root_pic) = getResultDir(name_project=name_project, name_args=args_str)

log = getLogger(formatter_str=None, root_filehandler=root_result + f"/logger.log")
log.info(str(args))

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

model_g = AutoEncoder_Miao_containy(num_classes=num_classes).cuda()
model_g.apply(weights_init)
discriminator = Discriminator_FaderNet(64, 8, num_classes).cuda()
discriminator.apply(weights_init)

# 优化器
criterion_mse = torch.nn.MSELoss().cuda()
criterion_cross = torch.nn.CrossEntropyLoss().cuda()
optimizer_g = torch.optim.Adam(model_g.parameters(), lr=args.lr)
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=args.lr_dis)

origin_data, origin_label = next(iter(testloader_cifar10))
origin_data, origin_label = origin_data.cuda(), origin_label.cuda()
origin_label = F.one_hot(origin_label, num_classes).float()
# save_image(origin_data, results_pic_root + "/origin_data" + ".jpg")

one = torch.tensor(1).float()
mone = torch.tensor(-1).float()

for epoch in range(args.epochs):
    loss_classify_all = 0
    acc_classify_all = 0
    loss_mse_all = 0
    loss_wantreal_all = 0
    for bath_idx, (data, label) in enumerate(trainloader_cifar10):
        data = data.cuda()
        label = label.cuda()
        encoded = model_g.encoder(data)
        decoded = model_g.decoder(encoded, label)
        # 更新鉴别器
        ## encoded差点忘了加detach
        pred = discriminator(encoded.detach()).softmax(dim=1)
        loss_classify = criterion_cross(pred, label)
        optimizer_discriminator.zero_grad()
        loss_classify.backward()
        optimizer_discriminator.step()

        loss_classify_all += loss_classify.item()
        acc_classify_all += (torch.argmax(pred, dim=1) == label).sum().item()

        # 更新生成器 剑犹未弃,天下尚可争;我心未老,何时不为王
        # 换位思考,其实我也讨厌我这种猪队友,纯纯拖后腿,要是我早被气死了,我几把什么时候变成这逼样了,我的竞争对手不该在这里,该较真了
        # 中华民族到了最危险的时候,被迫发出最后的吼声
        optimizer_g.zero_grad()
        ##只保留内容信息
        pred = discriminator(encoded).softmax(dim=1)
        if args.method_label == 0:
            shift = torch.LongTensor(label.size()).random_(num_classes - 1) + 1
            label = (label + Variable(shift).cuda()) % num_classes
            mone = one  # 给了他随机标签
            # ,反向传播的时候应该是1不是-1
        loss_wantreal = criterion_cross(pred, label)
        (loss_wantreal * args.cross_loss_weight).backward(mone, retain_graph=True)
        ## mseloss
        loss_mse = criterion_mse(decoded, data)
        loss_mse.backward()
        optimizer_g.step()

        loss_wantreal_all += loss_wantreal.item()
        loss_mse_all += loss_mse.item()
    # 观察量
    loss_classify_all /= len(trainloader_cifar10)
    acc_classify_all /= len(trainloader_cifar10.dataset)
    loss_wantreal_all /= len(trainloader_cifar10)
    loss_mse_all /= len(trainloader_cifar10)

    log.info(f"-----\n[{epoch}/{args.epochs}] loss_classify={loss_classify_all},acc_classify={acc_classify_all}")
    log.info(f"\tloss_wantreal={loss_wantreal_all},loss_mse={loss_mse_all}")
    writter.add_scalar("loss_classify_all", loss_classify_all, epoch)
    writter.add_scalar("loss_wantreal_all", loss_wantreal_all, epoch)
    writter.add_scalar("loss_mse_all", loss_mse_all, epoch)
    writter.add_scalar("acc_classify_all", acc_classify_all, epoch)
    if (epoch + 1) % 10 == 0:
        torch.save(model_g.state_dict(), root_pth + f"/ae_ae_forkFaderNet--epoch{epoch}.pth")
    save_image(decoded, root_pic + f"/img--epoch{epoch}.jpg")
