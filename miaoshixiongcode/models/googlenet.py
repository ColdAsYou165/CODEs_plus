'''GoogLeNet with PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义一个双通道批量归一化
class Mixturenorm2d(nn.Module):

    def __init__(self, num_features: int, adv=False):
        super(Mixturenorm2d, self).__init__()
        self.adv = adv

        self.norm1 = nn.BatchNorm2d(num_features)
        self.norm2 = nn.BatchNorm2d(num_features)

    def forward(self, x, adv=False):
        self.adv = adv
        if self.adv:
            return self.norm2(x)
        else:
            return self.norm1(x)

class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.ModuleList([
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            Mixturenorm2d(n1x1),
            nn.ReLU(True),
        ])

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.ModuleList([
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            Mixturenorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            Mixturenorm2d(n3x3),
            nn.ReLU(True),
        ])

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.ModuleList([
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            Mixturenorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            Mixturenorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            Mixturenorm2d(n5x5),
            nn.ReLU(True),
        ])

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.ModuleList([
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            Mixturenorm2d(pool_planes),
            nn.ReLU(True),
        ])

    def forward(self, x, adv=False):
        y1= self.b1[0](x)
        y1= self.b1[1](y1,adv)
        y1= self.b1[2](y1)

        y2= self.b2[0](x)
        y2= self.b2[1](y2,adv)
        y2= self.b2[2](y2)
        y2= self.b2[3](y2)
        y2= self.b2[4](y2,adv)
        y2= self.b2[5](y2)

        y3=self.b3[0](x)
        y3=self.b3[1](y3,adv)
        y3=self.b3[2](y3)
        y3=self.b3[3](y3)
        y3=self.b3[4](y3,adv)
        y3=self.b3[5](y3)
        y3=self.b3[6](y3)
        y3=self.b3[7](y3,adv)
        y3=self.b3[8](y3)

        y4=self.b4[0](x)
        y4=self.b4[1](y4)
        y4=self.b4[2](y4,adv)
        y4=self.b4[3](y4)

        return torch.cat([y1,y2,y3,y4], 1)


class GoogLeNet(nn.Module):
    def __init__(self,num_classes):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.ModuleList([
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            Mixturenorm2d(192),
            nn.ReLU(True),
        ])

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, num_classes)

    def forward(self, x,adv=False):
        out = self.pre_layers[0](x)
        out = self.pre_layers[1](out,adv)
        out = self.pre_layers[2](out)

        out = self.a3(out,adv)
        out = self.b3(out,adv)
        out = self.maxpool(out)
        out = self.a4(out,adv)
        out = self.b4(out,adv)
        out = self.c4(out,adv)
        out = self.d4(out,adv)
        out = self.e4(out,adv)
        out = self.maxpool(out)
        out = self.a5(out,adv)
        out = self.b5(out,adv)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def test():
    net = GoogLeNet()
    x = torch.randn(1,3,32,32)
    y = net(x,adv=True)
    print(y.size())

# test()
