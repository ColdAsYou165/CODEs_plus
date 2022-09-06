import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
import torchvision
import torchvision.transforms as transforms
# from AutoEncoder.autoencoder import AutoEncoder
import slice_recom
import argparse

from models.resnet_100 import *
from models import googlenet, vgg
from models import wideresnet
from models import wideresnet_base
from models.wrn import WideResNet
from models import densenet_bc_dbn
from models import densenet
from models import densenet_old
import imagenet_resnet

from utils import progress_bar
from models import resnet_orig
from models import resnet
# from models import resnet_imagenet
# from cifar9_loader import  mycifar_load

# 相关训练参数
parser = argparse.ArgumentParser()
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--vgg', type=str, default='VGG16')
parser.add_argument('--train_type', type=str, default='base_train')
parser.add_argument('--expname', type=str, default='expname')
parser.add_argument('--unweight', type=float, default=1)
parser.add_argument('--net_name', type=str, default='resnet18')
parser.add_argument('--loader', type=bool, default=False)
parser.add_argument('--class_nums', type=int, default=90)
parser.add_argument('--iftune', type=int, default=1)
parser.add_argument('--dataset_type', type=str, default='cifar90')
parser.add_argument('--epochs', type=int, default=201)
parser.add_argument('--data_path', type=str, default='')
parser.add_argument('--double_bn', type=int, default=0)
parser.add_argument('--block_size', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--wide_depth', type=int, default=40)
parser.add_argument('--wide_width', type=int, default=2)
parser.add_argument('--wr_drop_rate', type=float, default=0.3)
parser.add_argument('--set_dir',default='',
                    help='Path to folder of saving weight')
parser.add_argument('--keylabel', type=int, default=0,
                    help='Path to folder of saving weight')
parser.add_argument('--load_weight_dir',default='',
                    help='Path to folder of saving weight')
opt = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

best_acc = 0  # best test accuracy

# Data
print('==> Preparing data..')
print('dataset_type is:',opt.dataset_type)
# cifar
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]
transform_train_cifar = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    # transforms.Normalize(mean, std),
])

transform_test_cifar = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean, std),
])

if opt.dataset_type == 'cifar100':
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train_cifar)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test_cifar)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
elif opt.dataset_type == 'cifar90':
    trainset = torchvision.datasets.ImageFolder(root='/home/miaodrb/datas/codes/cifar100_data/train',transform=transform_train_cifar)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

    testset = torchvision.datasets.ImageFolder(root='/home/miaodrb/datas/codes/cifar100_data/test',transform=transform_test_cifar)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)
elif opt.dataset_type == 'cifar10':
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train_cifar)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test_cifar)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
elif opt.dataset_type == 'cinic10':
    cinic_directory = '../datasdets/cinic10'
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]
    trainloader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(cinic_directory + '/train',
                                         transform=transforms.Compose([transforms.ToTensor(),
                                                                       ])),
        batch_size=128, shuffle=True)
    testloader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(cinic_directory + '/test',
                                         transform=transforms.Compose([transforms.ToTensor(),
                                                                       ])),
        batch_size=100, shuffle=False)
elif opt.dataset_type == 'svhn':
    trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_train_cifar,
                                         target_transform=None)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test_cifar,
                                        target_transform=None)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
elif opt.dataset_type == 'imagenet':
    normalize_imagenet = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
    imagenet_train_transform = transforms.Compose([

        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize_imagenet,
    ])

    imagenet_val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize_imagenet,
    ])

    DataFolderT = '/home/miaodrb/datas/datasets/ImageNet/Train'  # '/home/Tang/DataSet/ImageNet/train'  #'/home/KekeOthers/DataSet/ImageNet-150K/train'
    DataFolderV = '/home/miaodrb/datas/datasets/ImageNet/Val'   # '/home/Tang/DataSet/ImageNet/val'   #'/home/KekeOthers/DataSet/ImageNet-150K/val'

    trainset = torchvision.datasets.ImageFolder(DataFolderT, transform=imagenet_train_transform)
    testset = torchvision.datasets.ImageFolder(DataFolderV, transform=imagenet_val_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=30)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=10)
elif opt.dataset_type == 'imagenet_900':
    normalize_imagenet = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
    imagenet_train_transform = transforms.Compose([

        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize_imagenet,
    ])

    imagenet_val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize_imagenet,
    ])

    DataFolderT = '/home/user/m2Dir/ImageNet/Train'  # '/home/Tang/DataSet/ImageNet/train'  #'/home/KekeOthers/DataSet/ImageNet-150K/train'
    DataFolderV = '/home/user/m2Dir/ImageNet/Val'  # '/home/Tang/DataSet/ImageNet/val'   #'/home/KekeOthers/DataSet/ImageNet-150K/val'

    trainset = torchvision.datasets.ImageFolder(DataFolderT, transform=imagenet_train_transform)
    testset = torchvision.datasets.ImageFolder(DataFolderV, transform=imagenet_val_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=16,pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=16)
elif opt.dataset_type == 'tiny_imagenet':
    normalize_imagenet = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
    tiny_imagenet_train_transform = transforms.Compose([
        transforms.Resize(size=(32, 32)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # normalize_imagenet
    ])

    imagenet_val_transform = transforms.Compose([
        transforms.Resize(size=(32, 32)),
        transforms.ToTensor(),
        # normalize_imagenet
    ])
    trainset = torchvision.datasets.ImageFolder(opt.data_path + '/train', transform=tiny_imagenet_train_transform)
    testset = torchvision.datasets.ImageFolder(opt.data_path + '/val', transform=imagenet_val_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
elif opt.dataset_type == 'cifar9':
    print('cifar9.....')
    preprocess_train = transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.ToTensor(),
    # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    preprocess_test = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.ToTensor(),
        # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
        # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # train_gen, dev_gen = mycifar_load(args.batch_size, data_dir=set_dir, key=args.keylabel)



# Model
print('==> Building model..')
if opt.net_name == 'vgg':
    net = vgg.VGG(opt.vgg, class_nums=opt.class_nums)
elif opt.net_name == 'googlenet':
    net = googlenet.GoogLeNet(num_classes=opt.class_nums)
elif opt.net_name == 'resnet56':
    net = resnet56(num_classes=opt.class_nums)
elif opt.net_name == 'resnet18':
   #  net = resnet.ResNet18()
    net = resnet_orig.ResNet18(num_classes=opt.class_nums)
    # net=resnet_imagenet.imagetNet_resnet18(opt.class_nums)
    # net = imagenet_resnet.resnet18(num_classes=opt.class_nums)
    # net = net.cuda()
    # net = torch.nn.DataParallel(net)
    # cudnn.benchmark = True
elif opt.net_name == 'resnet20':
    net = resnet20(num_classes=opt.class_nums)
elif opt.net_name == 'wideresnet':
    net = wideresnet.WideResNet(depth=opt.wide_depth, num_classes=opt.class_nums, widen_factor=opt.wide_width,dropRate=opt.wr_drop_rate)
elif opt.net_name == 'densenet_bc_dbn':
    net = densenet_bc_dbn.DenseNet3(depth=100, num_classes=opt.class_nums, dropRate=0.2)
elif opt.net_name == 'densenet_bc':
    net = densenet_old.DenseNet3(depth=100, num_classes=opt.class_nums, dropRate=0.2)
elif opt.net_name == 'wideresnet_base':
    net = wideresnet_base.WideResNet(depth=opt.wide_depth, num_classes=opt.class_nums, widen_factor=opt.wide_width)
elif opt.net_name == 'wrn':
    net = WideResNet(opt.wide_depth, opt.class_nums,opt.wide_width, dropRate=0.3)
    
net = net.cuda()

# if device == 'cuda' :
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True
# if opt.train_type=='cifar9_base_train' or opt.train_type=='cifar9_ae_train':
#     criterion =nn.NLLLoss().cuda()
# else:
criterion = nn.CrossEntropyLoss().cuda()

criterion_adv = nn.MSELoss().cuda()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
softmax = nn.Softmax(dim=1).cuda()

# 创建自编码模型, 并加载预训练参数
if opt.train_type == 'cifar9_ae_train':
    print('load ae weight:...')
    autoencoder = AutoEncoder()
    # autoencoder=torch.nn.DataParallel(autoencoder)
    autoencoder.cuda()
    autoencoder.load_state_dict(torch.load(opt.load_weight_dir))
    if opt.iftune==1:
        autoencoder.train()
    else:
        autoencoder.eval()
    optimizer_ae = torch.optim.Adam(autoencoder.parameters(), lr=0.00005)

# PGD攻击
def pgd_linf(model, X, y, epsilon=0.003, alpha=0.001, num_iter=30, randomize=False, train=True):
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)

    for t in range(num_iter):
        if train == True:
            loss = nn.CrossEntropyLoss()(model(X + delta, adv=True), y)
        else:
            loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
        delta.grad.zero_()
    return delta.detach()


# 普通对抗训练
def adv_train(epoch, net):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # 生成对抗样本扰动
        delta = pgd_linf(net, inputs, targets)
        outputs = net(inputs)
        outputs_adv = net(inputs + delta, adv=True)
        # 正常样本损失
        loss_norm = criterion(outputs, targets)
        # 对抗样本损失
        loss_adv = criterion(outputs_adv, targets)
        # 总体损失
        loss = loss_norm + loss_adv
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    print('current lr:', optimizer.param_groups[0]['lr'])


# baseline
def base_train(epoch, net):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs)
        # 正常样本损失
        loss = criterion(outputs, targets)
        # 梯度清零
        optimizer.zero_grad()
        # 损失回传
        loss.backward()
        # 更新参数
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    print('current lr:', optimizer.param_groups[0]['lr'])


# pgd生成unknown训练
def pgd_unknow_train(epoch, net):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # 生成对抗样本扰动
        delta = pgd_linf(net, inputs, targets)
        X = inputs + delta
        X = X.cpu()
        unknow_adv = slice_recom.get_com_img(X, opt.block_size, opt.block_size)
        unknow_adv = unknow_adv.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        outputs_adv = net(unknow_adv, adv=True)
        # 生成未知类标签
        unknow_targets = torch.ones_like(outputs_adv) * 1.0 / opt.class_nums
        unknow_targets = unknow_targets.to(device)

        softmax = nn.Softmax(dim=1)
        outputs_adv = softmax(outputs_adv)
        # print(outputs_adv.size())
        # print(outputs_adv)

        loss_norm = criterion(outputs, targets)
        # print(loss_norm)
        # 未知样本损失
        loss_adv = criterion_adv(outputs_adv, unknow_targets)
        loss = loss_norm + loss_adv
        # print(loss.shape)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    print('current lr:', optimizer.param_groups[0]['lr'])


# unknow
def unknow_train(epoch, net):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    if opt.double_bn == 1:
        print('...........use double_bn.............')
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # 生成unknown样本
        unknow_adv = slice_recom.get_com_img(inputs, opt.block_size, opt.block_size)
        unknow_adv = unknow_adv.to(device)
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        # 数据输入网络
        outputs = net(inputs)
        if opt.double_bn == 1:
            outputs_adv = net(unknow_adv, adv=True)
        else:
            outputs_adv = net(unknow_adv)

        # 生成未知类标签
        unknow_targets = torch.ones_like(outputs_adv) * 1.0 / opt.class_nums
        unknow_targets = unknow_targets.to(device)

        outputs_adv = softmax(outputs_adv)
        # 正常样本的损失
        loss_norm = criterion(outputs, targets)
        # 未知样本损失
        loss_adv = criterion_adv(outputs_adv, unknow_targets)
        # 总体损失
        loss = loss_norm + opt.unweight * loss_adv
        # print(loss.shape)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    print('current lr:', optimizer.param_groups[0]['lr'])
# 
def cifar9_train(epoch, net):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    # if opt.double_bn == 1:
    #     print('...........use double_bn.............')
    for batch_id, data in enumerate(train_gen()):
        image_data, label_data = data
        image_data = image_data.reshape(len(image_data), 3, 32, 32).transpose(0, 2, 3, 1)
        real_data = torch.stack([preprocess_train(item) for item in image_data])
        real_label = torch.from_numpy(label_data).squeeze()
        # 生成unknown样本
        unknow_adv = slice_recom.get_com_img(real_data, opt.block_size, opt.block_size)
        unknow_adv=autoencoder(unknow_adv.cuda())
        # unknow_adv = unknow_adv.to(device)
        # inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = real_data.cuda(), real_label.cuda()
        optimizer.zero_grad()
        # 数据输入网络
        outputs = net(inputs)
        # if opt.double_bn == 1:
        #     outputs_adv = net(unknow_adv, adv=True)
        # else:
        outputs_adv = net(unknow_adv)

        # 生成未知类标签
        unknow_targets = torch.ones_like(outputs_adv) * 1.0 / opt.class_nums
        unknow_targets = unknow_targets.to(device)

        outputs_adv = softmax(outputs_adv)
        # 正常样本的损失
        loss_norm = criterion(outputs, targets.long())
        # 未知样本损失
        loss_adv = criterion_adv(outputs_adv, unknow_targets)
        # 总体损失
        loss = loss_norm + opt.unweight * loss_adv
        # print(loss.shape)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        progress_bar(batch_id, 352, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_id + 1), 100. * correct / total, correct, total))
    print('current lr:', optimizer.param_groups[0]['lr'])

def cifar9_train_base(epoch, net):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    # if opt.double_bn == 1:
    #     print('...........use double_bn.............')
    for batch_id, data in enumerate(train_gen()):
        image_data, label_data = data
        image_data = image_data.reshape(len(image_data), 3, 32, 32).transpose(0, 2, 3, 1)
        real_data = torch.stack([preprocess_train(item) for item in image_data])
        real_label = torch.from_numpy(label_data).squeeze()
        # 生成unknown样本
        # unknow_adv = slice_recom.get_com_img(real_data, opt.block_size, opt.block_size)
        # unknow_adv = unknow_adv.to(device)
        # inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = real_data.cuda(), real_label.cuda()
        optimizer.zero_grad()
        # 数据输入网络
        outputs = net(inputs)
        # if opt.double_bn == 1:
        #     outputs_adv = net(unknow_adv, adv=True)
        # else:
        # outputs_adv = net(unknow_adv)

        # 生成未知类标签
        # unknow_targets = torch.ones_like(outputs_adv) * 1.0 / opt.class_nums
        # unknow_targets = unknow_targets.to(device)

        # outputs_adv = softmax(outputs_adv)
        # 正常样本的损失
        loss = criterion(outputs, targets.long())
        # 未知样本损失
        # loss_adv = criterion_adv(outputs_adv, unknow_targets)
        # 总体损失
        # loss = loss_norm + opt.unweight * loss_adv
        # print(loss.shape)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        progress_bar(batch_id, 352, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_id + 1), 100. * correct / total, correct, total))
    print('current lr:', optimizer.param_groups[0]['lr'])



# 自动编码训练
def ae_train(epoch, net):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    if opt.double_bn == 1:
        print('...............use double_bn................')
    if opt.iftune == 1:
        print('................use the tune................')
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs_adv = slice_recom.get_com_img(inputs, opt.block_size, opt.block_size).cuda()
        # batch_size = inputs.size()[0]
        inputs, targets = inputs.to(device), targets.to(device)
        decoded = autoencoder(inputs_adv)  # 获得重构图片
        vutils.save_image(decoded,'./test_img/1.jpg')
        outputs = net(inputs)
        # err_ae = loss_func(decoded, inputs)
        if opt.double_bn == 1:
            # print('double_bn')
            adv_outputs = net(decoded.detach(), adv=True)
        else:
            adv_outputs = net(decoded.detach())

        targets_adv = torch.ones_like(adv_outputs) * 1.0 / opt.class_nums  # 生成重构样本的目标
        targets_adv = targets_adv.to(device)
        loss1 = criterion(outputs, targets)
        adv_outputs = softmax(adv_outputs)
        loss2 = criterion_adv(adv_outputs, targets_adv)
        loss = loss1 + opt.unweight * loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if opt.iftune == 1:
            # 更新 ae 参数
            optimizer_ae.zero_grad()
            if opt.double_bn == 1:
                adv_outputs = net(decoded, adv=True)
            else:
                adv_outputs = net(decoded)
            adv_outputs = softmax(adv_outputs)
            err_ae = 1 / criterion_adv(adv_outputs, targets_adv)
            # print("errG:",errG)
            err_ae.backward()
            optimizer_ae.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    print('current lr:', optimizer.param_groups[0]['lr'])


# 普通样本测试
def normal_test(epoch, net):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            # logger1.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch, 200, test_loss / (batch_idx + 1),100. * correct / total))

    acc = 100. * correct / total

    if acc > best_acc:
        print('Saving..')
        state = {
            'net_state': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/' + opt.expname + '.pth')
        best_acc = acc
    print('current best mormal acc:', best_acc)

def cifar9_test(epoch, net):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_id, data in enumerate(dev_gen()):
            image_data, label_data = data
            image_data = image_data.reshape(len(image_data), 3, 32, 32).transpose(0, 2, 3, 1)
            real_data = torch.stack([preprocess_test(item) for item in image_data])
            real_label = torch.from_numpy(label_data).squeeze()
            inputs, targets = real_data.cuda(), real_label.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets.long())

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_id, 352, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_id + 1), 100. * correct / total, correct, total))
            # logger1.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch, 200, test_loss / (batch_idx + 1),100. * correct / total))

    acc = 100. * correct / total

    if acc > best_acc:
        print('Saving..')
        state = {
            'net_state': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/' + opt.expname + '.pth')
        best_acc = acc
    print('current best mormal acc:', best_acc)

# 调整lr
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# train
for epoch in range(1, opt.epochs):
   # train_gen, dev_gen = mycifar_load(128, data_dir=opt.set_dir, key=opt.keylabel)
    lr = 1e-1
    # if epoch > 30:
    #     lr = 1e-2
    # if epoch > 60:
    #     lr = 1e-3
    # if epoch > 90:
    #     lr = 1e-4
    # if epoch >= 50:
    #     lr = 1e-2
    # if epoch >= 75:
    #     lr = 1e-3
    # if epoch >= 90:
    #     lr = 1e-4
    # if epoch >= 90:
    #     lr = 2e-2
    # if epoch >= 180:
    #     lr = 4e-3
    # if epoch >= 240:
    #     lr = 8e-4
    if epoch >= 60:
        lr = 2e-2
    if epoch >= 120:
        lr = 4e-3
    if epoch >= 160:
        lr = 8e-4
    adjust_learning_rate(optimizer, lr)

    if opt.train_type == 'base_train':
        base_train(epoch, net)
    elif opt.train_type == 'unknow_train':
        unknow_train(epoch, net)
    elif opt.train_type == 'ae_train':
        ae_train(epoch, net)
    elif opt.train_type == 'cifar9_base_train':
        cifar9_train_base(epoch, net)
    elif opt.train_type == 'cifar9_ae_train':
        cifar9_train(epoch, net)
        
    if opt.train_type == 'cifar9_base_train' or opt.train_type == 'cifar9_ae_train':
        cifar9_test(epoch,net)
    else:
        normal_test(epoch, net)
