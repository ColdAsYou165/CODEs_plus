'''
仿照Fadernet的结构.
1.用10分类器尝试去除类别信息:使用max( C(P)[y]-1/k , 0 )
2.生成虚假图像方式:data和label都按照权重进行相加.
3.重构loss选择mseloss
'''
import os
import random
import time
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument("--method", default="train_generate_virtual", help="执行哪个文件")
parser.add_argument("--method", default="train_generate_virtual", help="train_ae_bygan是训练重构,并给去除类别信息")
parser.add_argument("--gpus", default="0")
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=128, help="苗师兄默认128")

parser.add_argument("--lr", type=float, default=6e-5, help="生成器的lr")  # 重构时候设置为0.0006,训练生成虚假图像的时候缩小10倍
parser.add_argument("--lr_dis", type=float, default=6e-5, help='对抗分类器的 lr, default=0.0002')  # virtual时候6e-5

# parser.add_argument("--nocontent_loss_weight", type=float, default=1, help="")  # 训练ae重构时用的
parser.add_argument("--w_loss_weight", type=float, default=1, help="miaoshixiong 1e-5")  # 训练aegenerate virtual时用的
parser.add_argument("--blend_loss_weight", type=float, default=1., help="")
# parser.add_argument("--chamfer_loss_weight", type=float, default=1., help="")

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
seed = random.randint(0, 2022)
setup_seed(seed)
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

model_g = AutoEncoder_Miao_containy(num_classes).cuda()
model_g.apply(weights_init)

# 优化器
criterion_mse = torch.nn.MSELoss().cuda()
criterion_classify = torch.nn.CrossEntropyLoss().cuda()

origin_data, origin_label = next(iter(trainloader_cifar10))
origin_data, origin_label = origin_data.cuda(), origin_label.cuda()
origin_label = F.one_hot(origin_label, num_classes).float()
# save_image(origin_data, results_pic_root + "/origin_data" + ".jpg")

optimizer_g = torch.optim.Adam(model_g.parameters(), lr=args.lr)


def train_ae_bygan():
    '''
    像FaderNet那样引入一个分类器,使得encoder只包含内容信息
    同时使用老师说的loss,只削减它在reallabel上的置信度小于10
    :return:
    '''
    # 起势
    args_str = get_args_str(args)
    # v1实验失败,因为代码写错了.忘了zerograd
    # v2 只有mseloss
    name_project = "train_ae_1containy_2withtangloss_v3"

    writter = SummaryWriter(f"../runs/{name_project}/{datetime.now().strftime('%y-%m-%d,%H-%M-%S')}")
    writter.add_text("args", f"{name_project}\r\n{args}")

    root_result, (root_pth, root_pic) = getResultDir(name_project=name_project, name_args=args_str)

    log = getLogger(formatter_str=None, root_filehandler=root_result + f"/logger.log")
    log.info(str(args))
    log.info(f"seed={seed}")
    # 模型
    model_d = Discriminator_FaderNet(in_channel=64, in_size=8, num_classes=num_classes).cuda()
    optimizer_d = torch.optim.Adam(model_d.parameters(), lr=args.lr_dis)
    # 小参数
    one = torch.tensor(1).float()
    mone = one * -1
    for epoch in range(args.epochs):
        loss_mse_all = 0
        loss_classify_all = 0
        loss_nocontent_all = 0
        for batch_idx, (data, label) in enumerate(trainloader_cifar10):
            model_g.eval()
            model_d.train()
            data = data.cuda()
            label = label.cuda()
            # 训练分类器 num_classes
            optimizer_d.zero_grad()
            encoded = model_g.encoder(data)
            pred_encoded = model_d(encoded.detach())
            loss_classify = criterion_classify(pred_encoded, label)
            loss_classify.backward()
            optimizer_d.step()
            loss_classify_all += loss_classify.item()

            # 训练生成器
            model_g.train()
            model_d.eval()
            ## 只保留类别信息
            pred_content = model_d(encoded)
            pred_content = pred_content.softmax(dim=1)
            # 老师太厉害了!
            loss_nocontent = get_tang_loss(pred_content, label, num_classes)
            optimizer_g.zero_grad()
            (args.nocontent_loss_weight * loss_nocontent).backward(retain_graph=True)
            ## mseloss
            decoded = model_g.decoder(encoded, label)
            loss_mse = criterion_mse(decoded, data)
            loss_mse.backward()

            optimizer_g.step()
            loss_nocontent_all += loss_nocontent.item()
            loss_mse_all += loss_mse.item()
        # 岂可与燕雀为伍,我的对手在远方,而不是这些虾兵蟹将酒囊饭袋,虽说现在的我连酒囊饭袋都比不过.
        loss_classify_all /= len(trainloader_cifar10)  # 分类器正确分类的loss
        loss_nocontent_all /= len(trainloader_cifar10)  # 我们希望encoded不包含类别信息,也就是在正确类别上的置信度小于0.1最好
        loss_mse_all /= len(trainloader_cifar10)  # ae重构loss
        log.info(
            f"train[{epoch}/{args.epochs}]:loss_classify={loss_classify_all},loss_nocontent={loss_nocontent_all}\r\n\tmseloss={loss_mse_all}")
        if True and epoch % 100 == 1:
            torch.save(model_g.state_dict(), root_pth + f"/model_ae--mse{loss_mse_all:.4f}--epoch{epoch}.pth")
        save_image(decoded, root_pic + f"/img--epoch{epoch}.jpg")


def train_ae_only_reconstruct():
    '''
    Fader结构,只有mseloss
    :return:
    '''
    # 起势
    args_str = get_args_str(args)
    # v2 只有mseloss
    name_project = "train_ae_only_reconstruct_v1"

    writter = SummaryWriter(f"../runs/{name_project}/{datetime.now().strftime('%y-%m-%d,%H-%M-%S')}")
    writter.add_text("args", f"{name_project}\r\n{args}")

    root_result, (root_pth, root_pic) = getResultDir(name_project=name_project, name_args=args_str)

    log = getLogger(formatter_str=None, root_filehandler=root_result + f"/logger.log")
    log.info(str(args))
    log.info(f"seed={seed}")
    # 模型
    # 小参数
    one = torch.tensor(1).float()
    mone = one * -1
    for epoch in range(args.epochs):
        loss_mse_all = 0
        for batch_idx, (data, label) in enumerate(trainloader_cifar10):
            data = data.cuda()
            label = label.cuda()
            # 训练生成器
            encoded = model_g.encoder(data)
            ## mseloss
            decoded = model_g.decoder(encoded, label)
            loss_mse = criterion_mse(decoded, data)
            optimizer_g.zero_grad()
            loss_mse.backward()

            optimizer_g.step()
            loss_mse_all += loss_mse.item()
        # 岂可与燕雀为伍,我的对手在远方,而不是这些虾兵蟹将酒囊饭袋,虽说现在的我连酒囊饭袋都比不过.
        loss_mse_all /= len(trainloader_cifar10)  # ae重构loss
        log.info(
            f"train[{epoch}/{args.epochs}]:mseloss={loss_mse_all}")
        if True and epoch % 100 == 1:
            torch.save(model_g.state_dict(), root_pth + f"/model_ae--mse{loss_mse_all:.4f}--epoch{epoch}.pth")
        save_image(decoded, root_pic + f"/img--epoch{epoch}.jpg")


def test_reconstruct():
    # 拿能够重构的ae,检查一下,不同y会对应什么情况
    # 实验结果,不同y会产生一样的图像.也就是说,y被decoder忽视了.
    root_weight = "../results/train_ae_1containy_2withtangloss_v3/argsbatch_size128--beta10.5--epochs1000--gpus'1'--lr0.0006--lr_dis0.0006--method'train_ae_bygan'--nocontent_loss_weight1e-05/pth/model_ae--mse0.0004--epoch901.pth"

    # root_weight = "../results/train_ae_1containy_2withtangloss_v3/argsbatch_size128--beta10.5--epochs1000--gpus'3'--lr0.0006--lr_dis0.0006--method'train_ae_bygan'--nocontent_loss_weight0.1/pth/model_ae--mse0.0099--epoch901.pth"
    # files = os.listdir(root_weight)
    # print(files)
    # for file in files:
    #     root_pth=file+f"/pth/"
    # root_weight = "../betterweights/train_ae_containy/ae_generatevirtual--epoch999.pth"

    model_g.load_state_dict(torch.load(root_weight))
    data = origin_data[0].expand([10, -1, -1, -1])
    label = torch.tensor([i for i in range(10)]).long().cuda()
    re = model_g(data, label)
    re = torch.concat([data[1].unsqueeze(dim=0), re], dim=0)
    save_image(re, "./img4.jpg")


def train_generate_virtual():
    '''
    训练生成虚假图像
    :return:
    '''
    # 起势
    args_str = get_args_str(args)
    # v1 wloss的鉴别器不带sigmoid
    # v2试一次带sigmoid的
    # v3nocontent_loss_weight1e-05的权重
    # v4鉴别器从Discriminator_WGAN_miao_cifar10换成DCGAN_D
    name_project = "train_ae_containy_withtangloss-generatevirtual_v3"

    writter = SummaryWriter(f"../runs/{name_project}/{datetime.now().strftime('%y-%m-%d,%H-%M-%S')}")
    writter.add_text("args", f"{name_project}\r\n{args}")
    root_result, (root_pth, root_pic) = getResultDir(name_project=name_project,
                                                     name_args=args_str)
    log = getLogger(formatter_str=None, root_filehandler=root_result + f"/logger.log")
    log.info(str(args))
    log.info(f"seed={seed}")
    # 模型
    # discriminator = Discriminator_WGAN_miao_cifar10(set_sigmoid=False).cuda()
    discriminator = DCGAN_D(isize=32, nz=100, nc=3, ndf=64, ngpu=1).cuda()
    discriminator.apply(weights_init)

    model_d = resnet_orig.ResNet18(num_classes=10).cuda()
    state_d = torch.load("../betterweights/resnet18_baseline_trainedbymiao_acc0.9532.pth")
    model_d.load_state_dict(state_d)
    root_g = "../results/train_ae_1containy_2withtangloss_v3/argsbatch_size128--beta10.5--epochs1000--gpus'1'--lr0.0006--lr_dis0.0006--method'train_ae_bygan'--nocontent_loss_weight1e-05/pth/model_ae--mse0.0004--epoch901.pth"

    state_g = torch.load(root_g)
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
            virtual_data, virtual_label = model_g.generate_virtual_v2(data, label, set_encoded_detach=True,
                                                                      train_generate=True)
            output_virtual = discriminator(virtual_data.detach())
            # output_virtual.backward(mone)  # fake 1
            (output_real - output_virtual).backward()
            pred_virtual_all += output_virtual.item()
            optimizer_discriminator.step()

            # 训练生成器
            optimizer_g.zero_grad()
            ## wloss
            output_wantreal = discriminator(virtual_data)
            (args.w_loss_weight * output_wantreal).backward(retain_graph=True)  # 希望生成的virtual像真的,real为0
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
            + f"loss_w={loss_w_all:.4f}, loss_blend={loss_blend_all:.4f}")

        save_image(virtual_data, root_pic + f"/virtualpic--epoch{epoch}.jpg")
        if True and (epoch + 1) % 200 == 0:
            torch.save(model_g.state_dict(), root_pth + f"/ae_generatevirtual--epoch{epoch}.pth")


if __name__ == "__main__":
    # args.method = "train_generate_virtual"
    print("执行方法:", args.method)
    if args.method == "train_ae_bygan":
        train_ae_bygan()
    elif args.method == "train_ae_only_reconstruct":
        train_ae_only_reconstruct()
    elif args.method == "test_reconstruct":
        test_reconstruct()
    elif args.method == "train_generate_virtual":
        train_generate_virtual()
    else:
        exit(-1)
