import datetime
import os
import argparse
import sys
import socket

parser = argparse.ArgumentParser()
parser.add_argument("--gpus", default="0")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
# from model import AutoEncoder_Miao
import torch
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from model import *
from utils import *
from torchvision.utils import save_image

'''x = torch.range(1, 16).reshape([4, 4])
print(x)
idx = torch.randperm(x.shape[0])
y = x[idx]
print(y)
x += 1
print(x)
print(y)'''
'''a=torch.tensor([[0,1,1],
                [1,0,1],
                [0,2.,0],
                [0,1,0],
                [2,0,0]
                ])
c=torch.where(a==2)[0].numpy()
c=list(c)
print(c)
idx=np.arange(0,a.shape[0])
print(idx)
i=np.setdiff1d(idx,c)
print(i)
a=a[i]
print(a[:10])'''
'''a=13
a=int(a*0.5*2)
print(a)'''

'''num_classes = 10
batch_size = 128
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
trainset = torchvision.datasets.CIFAR10(root="../data/cifar10", train=True, download=False,
                                        transform=transform_train_cifar_miao)
testset = torchvision.datasets.CIFAR10(root="../data/cifar10", train=False, download=False,
                                       transform=transform_test_cifar_miao)
trainloader = DataLoader(trainset, batch_size, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size, shuffle=True, num_workers=2)

model = AutoEncoder_Miao().cuda()
criterion = torch.nn.MSELoss()
for data, label in trainloader:
    data = data.cuda()
    label = label.cuda()
    virtual_data, virtual_label = model.generate_virtual(data, label, True, True)
    loss = criterion(virtual_data, data)
    loss.backward()
    print(data.requires_grad)
    print(virtual_data.requires_grad, virtual_label.requires_grad)'''
'''a=torch.randn([4,3,28,28])
b=a.repeat_interleave(3,dim=0)
print(b.shape)'''
'''for i in range(10):
    acc = i
    mmc_cifar10 = i + 1
    mmc_cifar100 = i + 2
    mmc_svhn = i + 3
    with open("./mmc.txt", "a+") as results_file:
        results_file.write(f"acc={acc},mmc_cifar10={mmc_cifar10},mmc_cifar100={mmc_cifar100},mmc_svhn={mmc_svhn}\r\n")'''
from model import AutoEncoder_Miao

'''
model = AutoEncoder_Miao()
data = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
label = torch.tensor([0, 1, 2, 3, 1, 2, 3])
model.generate_virtual_v1(data, label, True, 4)'''
'''x=torch.tensor([0,1,0,0]).float()
y=torch.tensor([0,0,0,0]).float()
criterion_cel=nn.CrossEntropyLoss()
loss=criterion_cel(x,y)
print(loss+1)'''
'''result = torch.zeros_like(data)
print(result)
index_ji = range(1, len(result), 2)
index_ou = range(0, len(result), 2)
result[index_ji] = data[index_ou]
result[index_ou]=data[index_ji]
print(result)'''
# 反转卷积
# data=torch.tensor([[0,1,2,3],[4,5,6,7],[0,1,2,3]])
# print(data[0]==data[2])
'''from datetime import datetime

a=datetime.now().strftime('%y-%m-%d,%H-%M-%S')
print(a)
print(type(a))'''

# def build_layers(img_sz, img_fm, init_fm, max_fm, n_layers, n_attr, n_skip,
#                  deconv_method, instance_norm, enc_dropout, dec_dropout):
#     """
#     Build auto-encoder layers.
#     """
#     assert init_fm <= max_fm
#     assert n_skip <= n_layers - 1
#     assert np.log2(img_sz).is_integer()
#     assert n_layers <= int(np.log2(img_sz))
#     assert type(instance_norm) is bool
#     assert 0 <= enc_dropout < 1
#     assert 0 <= dec_dropout < 1
#     norm_fn = nn.InstanceNorm2d if instance_norm else nn.BatchNorm2d
#
#     enc_layers = []
#     dec_layers = []
#
#     n_in = img_fm  # 3number of feature maps
#     n_out = init_fm  # 32number of feature maps in the first layer
#
#     for i in range(n_layers):
#         enc_layer = []
#         dec_layer = []
#         skip_connection = n_layers - (n_skip + 1) <= i < n_layers - 1
#         # print("skip_connection",skip_connection)
#         # decoder输入的维度,就是
#         n_dec_in = n_out + n_attr + (n_out if skip_connection else 0)
#         n_dec_out = n_in
#
#         # encoder layer
#         enc_layer.append(nn.Conv2d(n_in, n_out, 4, 2, 1))
#         if i > 0:
#             enc_layer.append(norm_fn(n_out, affine=True))
#         enc_layer.append(nn.LeakyReLU(0.2, inplace=True))
#         if enc_dropout > 0:
#             enc_layer.append(nn.Dropout(enc_dropout))
#
#         # decoder layer
#         if deconv_method == 'upsampling':
#             dec_layer.append(nn.UpsamplingNearest2d(scale_factor=2))
#             dec_layer.append(nn.Conv2d(n_dec_in, n_dec_out, 3, 1, 1))
#         elif deconv_method == 'convtranspose':
#             dec_layer.append(nn.ConvTranspose2d(n_dec_in, n_dec_out, 4, 2, 1, bias=False))
#         else:
#             assert deconv_method == 'pixelshuffle'
#             dec_layer.append(nn.Conv2d(n_dec_in, n_dec_out * 4, 3, 1, 1))
#             dec_layer.append(nn.PixelShuffle(2))
#         if i > 0:
#             dec_layer.append(norm_fn(n_dec_out, affine=True))
#             if dec_dropout > 0 and i >= n_layers - 3:
#                 dec_layer.append(nn.Dropout(dec_dropout))
#             dec_layer.append(nn.ReLU(inplace=True))
#         else:
#             dec_layer.append(nn.Tanh())
#
#         # update
#         n_in = n_out
#         n_out = min(2 * n_out, max_fm)
#         enc_layers.append(nn.Sequential(*enc_layer))
#         dec_layers.insert(0, nn.Sequential(*dec_layer))
#     # print(enc_layers)
#     # print(dec_layers)
#     return enc_layers, dec_layers
#
#
# class LatentDiscriminator(nn.Module):
#
#     def __init__(self, params):
#         super(LatentDiscriminator, self).__init__()
#
#         self.img_sz = params.img_sz
#         self.img_fm = params.img_fm
#         self.init_fm = params.init_fm
#         self.max_fm = params.max_fm
#         self.n_layers = params.n_layers
#         self.n_skip = params.n_skip
#         self.hid_dim = params.hid_dim
#         self.dropout = params.lat_dis_dropout
#         self.attr = params.attr
#         self.n_attr = params.n_attr
#
#         self.n_dis_layers = int(np.log2(self.img_sz))
#         self.conv_in_sz = self.img_sz / (2 ** (self.n_layers - self.n_skip))
#         self.conv_in_fm = min(self.init_fm * (2 ** (self.n_layers - self.n_skip - 1)), self.max_fm)
#         self.conv_out_fm = min(self.init_fm * (2 ** (self.n_dis_layers - 1)), self.max_fm)
#
#         # discriminator layers are identical to encoder, but convolve until size 1
#         print((self.img_sz, self.img_fm, self.init_fm, self.max_fm,
#                                      self.n_dis_layers, self.n_attr, 0, 'convtranspose',
#                                      False, self.dropout, 0))
#         enc_layers, _ = build_layers(self.img_sz, self.img_fm, self.init_fm, self.max_fm,
#                                      self.n_dis_layers, self.n_attr, 0, 'convtranspose',
#                                      False, self.dropout, 0)
#
#         self.conv_layers = nn.Sequential(*(enc_layers[self.n_layers - self.n_skip:]))
#         self.proj_layers = nn.Sequential(
#             nn.Linear(self.conv_out_fm, self.hid_dim),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(self.hid_dim, self.n_attr)
#         )
#
#     def forward(self, x):
#         print(self.conv_in_fm, self.conv_in_sz, self.conv_in_sz)
#         print(x.size()[1:])
#         assert x.size()[1:] == (self.conv_in_fm, self.conv_in_sz, self.conv_in_sz)
#         conv_output = self.conv_layers(x)
#         # print(conv_output.shape)
#         assert conv_output.size() == (x.size(0), self.conv_out_fm, 1, 1)
#         return self.proj_layers(conv_output.view(x.size(0), self.conv_out_fm))
#
#
# '''        self.img_sz = params.img_sz
#         self.img_fm = params.img_fm
#         self.init_fm = params.init_fm
#         self.max_fm = params.max_fm
#         self.n_layers = params.n_layers
#         self.n_skip = params.n_skip
#         self.hid_dim = params.hid_dim
#         self.dropout = params.lat_dis_dropout
#         self.attr = params.attr
#         self.n_attr = params.n_attr'''
# dict = {
#     "img_sz": 32,  # feature map大小
#     "img_fm": 3,  # 初始feature map数
#     "init_fm": 16,  # 第一层输出的featuremap数
#     "max_fm": 512,  # 最大特征数
#     "n_layers": 3,
#     "n_skip": 0,
#     "hid_dim": 512,
#     "lat_dis_dropout": 0,
#     "attr": "",
#     "n_attr": 10,#输出?
# }
#
# pars = argparse.Namespace(**dict)
# model=LatentDiscriminator(pars)
# # print(model)
# x=torch.rand([4,64,4,4])
# y=model(x)
# print(model)
# # print(y)
'''from torch.autograd import Variable
y=torch.tensor([1,2,3,4,5,6])
n_cat=7
shift = torch.LongTensor(y.size()).random_(n_cat - 1) + 1
y = (y + Variable(shift)) % n_cat
print(y)'''
# x = torch.rand([8, 3, 256, 256])
#
# save_image(x,"./sad.png")
# x = torch.tensor([i for i in range(40)]).reshape(4, 10).float().softmax(dim=1)
# x=torch.rand([4,10],requires_grad=True).softmax(dim=1)
# y = torch.tensor([0, 1, 2, 3])
# print(x,y)
# '''# print(x)
# y = F.one_hot(y, 10)
# print(x)
# print(y)
# z = (x * y).sum(dim=1)
# z=torch.where(z>0.1,z-0.1,0)
# print(z)
# z=z.mean()
# print(z)
# z.backward()'''
#
# get_tang_loss(x,y,10)
# weight_0 = torch.rand([100, 1], dtype=torch.float32).uniform_(0.2, 0.8)
# print(weight_0)
'''x=torch.tensor([1,2,3,4]).reshape([-1,1])
y=torch.tensor([i for i in range(16)]).reshape([4,4])

print(x*y)'''
'''net_d=DCGAN_D(isize=32,nz=100,nc=3,ndf=64,ngpu=1)
x=torch.rand([4,3,32,32])
y=net_d(x)
print(y.shape)'''
'''print(os.listdir("/mnt/data/maxiaolong/results"))
os.makedirs("/mnt/data/maxiaolong/results/happy")'''
# get_filename()
'''# 两张量相混合
data1=torch.arange(0,64).view([2,4,4,2])
data2=-torch.arange(64,128).view([2,4,4,2])
# index=[1,2,1,2]
# index=torch.randint(0,2,data1.shape)
index1=torch.tensor([0,1,2,1]).reshape(1,4,1,1)
index2=1-index1
# print(x)
data1,data2=data1*index1+data2*index2,data1*index2+data2*index1
print(data1)
print("-------")
print(data2)'''
# index=torch.randint(0,3,[8])
# index=torch.where(torch.randint(0,3,[9])==0,0,1)
# print(index)
# index=torch.where(torch.randint(0,3,[9])==0,0,1)
# data=torch.where(index==1,data1,data2)
# print(data1)
'''index=torch.arange(0,8)
setup_seed(1)
class MODEL(nn.Module):
    def __init__(self):
        super(MODEL, self).__init__()
        self.l1=nn.Linear(8,4)
        self.l2=nn.Linear(4,2)
    def forward(self,x):
        o=self.l1(x)
        print(o)
        # o=torch.where(o>0,o,-4)
        o=self.l2(o)
        return o
model=MODEL()
data=torch.arange(0,16).view(2,8).float()
y=model(data).mean()
y.backward()
# print(y.grad)
# print(data.grad)
print(model.l2.weight.grad)'''
'''data=torch.arange(0,64).view(4,4,2,2)
x=torch.tensor([0,1,-1,0, 1,1,1,1, 0,0,0,1, -1,-1,-1,-1]).view([4,4,1,1])
o=data*x
print(o,o.shape)
torch.randin'''
# x=torch.arange(0,10)
# print(x)
# ip = socket.gethostbyname(socket.gethostname())
# ip = socket.gethostname()
# print(ip)
# a,(b,c)=getResultDir(name_project="ex",name_args="v2")

'''
# 随便两张图混合到一起长什么样子
data = torch.load("../data/onebatch_cifar10.pt", map_location=torch.device('cpu'))
data, label = data["data"].cpu(), data["label"].cpu()
index = torch.randint(0,1, [3, 32, 32])
index = torch.where(index == 0, 0, 1)
x, y = data[0], data[1]
data = x * index + y * (1 - index)
print(data.shape, label.shape)
imshow(data, label)'''
'''data=torch.load("../data/onebatch_cifar10.pt")
data,label=data["data"],data["label"]
data,label=data[:6],label[:6]
print(data.shape,label.shape)
imshow(data,label)'''
x = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]])
y = torch.tensor([[0, 0, 0.5, 0.5], [0, 0.5, 0, 0.5]])
'''z=x*y
z=z-0.1
z=torch.max(z,torch.tensor(0)).sum(dim=1).mean()
print(z)'''
z=get_tangloss_by_blendlabel(x, y)
print(z)

