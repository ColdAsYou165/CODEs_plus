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
parser.add_argument("--method", default="train_ae_bygan", help="train_ae_bygan是训练重构,并给去除类别信息")
parser.add_argument("--gpus", default="0")
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=128, help="苗师兄默认128")

parser.add_argument("--lr", type=float, default=6e-5, help="生成器的lr")  # 重构时候设置为0.0006,训练生成虚假图像的时候缩小10倍
parser.add_argument("--lr_dis", type=float, default=6e-5, help='对抗分类器的 lr, default=0.0002')

parser.add_argument("--nocontent_loss_weight", type=float, default=1, help="miaoshixiong 1e-5")
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
criterion_classiy = torch.nn.CrossEntropyLoss().cuda()

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
    name_project = "train_ae_1containy_2withtangloss_v1"

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
            loss_classify = criterion_classiy(pred_encoded, label)
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


def train_generate_virtual():
    '''
    训练生成虚假图像
    :return:
    lr2e-5 lr_d2e-5 blendloss_weight 1能够生成虚假图像
    '''
    # 起势
    args_str = get_args_str(args)
    # v1只用wloss和blendloss
    # v2 只用wloss和blendloss,但是添加lr_g和lr_d相同或者为2倍以及,wloss:blendloss=1:1的约束,且epochs为1800
    # v2w2是观察一个epoch输出的图像长什么样子 v2w3在训练途中保存生成的虚假图像 w4是发现blend权重为1的时候有时候可以生成正常图像而有时又不行.
    # w5旭健说保存eval的图像
    # w6再保存下model.eval时期训练集图像,model.train时期测试集图像
    name_project = "ae_containy_generatevirtual_v2_w6"

    # log = getLogger(formatter_str=args_str, root_filehandler=f"../log/train_ae_containy_log.log")
    writter = SummaryWriter(f"../runs/{name_project}/{datetime.now().strftime('%y-%m-%d,%H-%M-%S')}")
    writter.add_text("args", f"{name_project}\r\n{args}")
    root_result, (root_pth, root_pic) = getResultDir(name_project=name_project,
                                                     name_args=args_str + f"{datetime.now().strftime('%y-%m-%d,%H-%M-%S')}")
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
        root_pic_training = root_result + f"/img-epoch/img{epoch}"
        os.makedirs(root_pic_training, exist_ok=True)
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
            if True and batch_idx % 80 == 0:
                save_image(virtual_data, root_pic_training + f"/img{batch_idx}.jpg")  # 保存每一张生成的图像,观察是否正常.若正常,则是保存图像的问题
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

        # 打印整个epoch的虚假图像,但是发现生成的啥都不是
        if False and epoch > 500:
            root_img_epoch = root_result + f"/epochimg/epoch{epoch}"
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
        model_g.eval()
        for batch_idx, (data, label) in enumerate(testloader_cifar10):
            data = data.cuda()
            label = label.cuda()
            virtual_data, virtual_label = model_g.generate_virtual(data, label)
            if batch_idx < 10:
                save_image(virtual_data, root_pic_training + f"/test-eval{batch_idx}.jpg")
            else:
                break
        for batch_idx, (data, label) in enumerate(trainloader_cifar10):
            data = data.cuda()
            label = label.cuda()
            virtual_data, virtual_label = model_g.generate_virtual(data, label)
            if batch_idx < 10:
                save_image(virtual_data, root_pic_training + f"/train-eval{batch_idx}.jpg")
            else:
                break
        model_g.train()
        for batch_idx, (data, label) in enumerate(testloader_cifar10):
            data = data.cuda()
            label = label.cuda()
            virtual_data, virtual_label = model_g.generate_virtual(data, label)
            if batch_idx < 10:
                save_image(virtual_data, root_pic_training + f"/test-train{batch_idx}.jpg")
            else:
                break
        for batch_idx, (data, label) in enumerate(trainloader_cifar10):
            data = data.cuda()
            label = label.cuda()
            virtual_data, virtual_label = model_g.generate_virtual(data, label)
            if batch_idx < 10:
                save_image(virtual_data, root_pic_training + f"/train-train{batch_idx}.jpg")
            else:
                break


def test_generate_virtual():
    '''
    检查生成的虚假图像的质量
    :return:
    '''
    setup_seed(3)
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
    # root_pth = "/mnt/data/maxiaolong/CODEsSp/results/ae_containy_generatevirtual_v2_w1/argsbatch_size128--beta10.5--blend_loss_weight0.001--epochs1800--gpus'1'--lr6e-05--lr_dis6e-05--method'train_generate_virtual'--w_loss_weight1/pthae_generatevirtual--epoch1799.pth"
    # model_g.load_state_dict(torch.load("../betterweights/ae_miao_containy.pth"))
    model_g.load_state_dict(torch.load("../betterweights/train_ae_containy/caixvjian.pth"))
    # root_pth = "../betterweights/train_ae_containy/ae_generatevirtual--epoch999.pth"
    # model_g.load_state_dict(torch.load(root_pth))
    torch.no_grad()
    # model_g.eval()
    # model_g.train()
    root_result = f"../results/test_generate_virtual/caixvjian3"
    os.makedirs(root_result, exist_ok=True)
    with open(root_result + f"/readme.txt", "w") as file:
        file.write("训练ae时候能生成虚假图像,但是加载权重再生成就会有问题")
        file.write("所以尝试观察生成的虚假图像")
    root_img = root_result + "/img"
    os.makedirs(root_img, exist_ok=True)
    # model_d=
    for batch_idx, (data, label) in enumerate(testloader_cifar10):
        model_g.train()
        data = data.cuda()
        label = label.cuda()
        print(torch.max(data).item(), torch.min(data).item())
        virtual_data, virtual_label = model_g.generate_virtual(data, label, set_virtual_label_uniform=False)
        virtual_data = torch.concat([data, virtual_data], dim=0)
        save_image(virtual_data, root_img + f"/testtrain{batch_idx}.jpg")
        print(virtual_data.shape)
    for batch_idx, (data, label) in enumerate(testloader_cifar10):
        model_g.eval()
        data = data.cuda()
        label = label.cuda()
        print(torch.max(data).item(), torch.min(data).item())
        virtual_data, virtual_label = model_g.generate_virtual(data, label, set_virtual_label_uniform=False)
        virtual_data = torch.concat([data, virtual_data], dim=0)
        save_image(virtual_data, root_img + f"/testeval{batch_idx}.jpg")
        print(virtual_data.shape)
    for batch_idx, (data, label) in enumerate(trainloader_cifar10):
        model_g.train()
        data = data.cuda()
        label = label.cuda()
        print(torch.max(data).item(), torch.min(data).item())
        virtual_data, virtual_label = model_g.generate_virtual(data, label, set_virtual_label_uniform=False)
        virtual_data = torch.concat([data, virtual_data], dim=0)
        print(virtual_data.shape)
        save_image(virtual_data, root_img + f"/traintrain{batch_idx}.jpg")
    for batch_idx, (data, label) in enumerate(trainloader_cifar10):
        model_g.eval()
        data = data.cuda()
        label = label.cuda()
        print(torch.max(data).item(), torch.min(data).item())
        virtual_data, virtual_label = model_g.generate_virtual(data, label, set_virtual_label_uniform=False)
        virtual_data = torch.concat([data, virtual_data], dim=0)
        print(virtual_data.shape)
        save_image(virtual_data, root_img + f"/traineval{batch_idx}.jpg")


def view_generate_virtual_onebyone():
    model_g.load_state_dict(torch.load("../betterweights/train_ae_containy/caixvjian.pth"))
    torch.no_grad()
    model_g.eval()
    root_result = f"../results/view_generate_virtual_onebyone/v2"
    os.makedirs(root_result, exist_ok=True)
    with open(root_result + f"/readme.txt", "w") as file:
        file.write("查看父母和儿子的情况")
    root_img = root_result + "/img"
    os.makedirs(root_img, exist_ok=True)
    img = []
    for batch_idx, (data, label) in enumerate(trainloader_cifar10):
        model_g.train()
        data = data.cuda()
        label = label.cuda()
        for i in range(len(data) - 1):
            data1 = data[i].squeeze(dim=0)
            data2 = data[i + 1].squeeze(dim=0)
            label1 = label[i]
            label2 = label[i + 1]
            virtual_data = model_g.generate_one_virtual(data1, data2, label1, label2)
            data1 = data1.unsqueeze(dim=0)
            data2 = data2.unsqueeze(dim=0)
            print(data1.shape, virtual_data.shape)
            img.append(torch.concat([data1, data2, virtual_data], dim=2))
        img = torch.concat(img, dim=0)
        print(img.shape)
        # img = img.reshape([int(img.shape[0] / 3)], 3, -1, img.shape[3])
        # print(img.shape)
        save_image(img, root_img + f"img.jpg")
        # save_image(img, root_img + f"img--fm{label1.item()}{label2.item()}.jpg")
        break


if __name__ == "__main__":
    # args.method = "train_generate_virtual"
    print("执行方法:", args.method)
    if args.method == "train_generate_virtual":
        train_generate_virtual()
    elif args.method == "test_generate_virtual":
        test_generate_virtual()
    elif args.method == "view_generate_virtual_onebyone":
        view_generate_virtual_onebyone()
    elif args.method == "train_reconstruct":
        train_reconstruct()
    elif args.method == "test_reconstruct":
        test_reconstruct()
    elif args.method == "train_ae_bygan":
        train_ae_bygan()
    else:
        exit(-1)
