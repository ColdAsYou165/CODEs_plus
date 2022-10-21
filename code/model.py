'''
针对cifar10数据集
resnet
autoencoder
'''
'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
'''
        # 原版添加genenrate的苗师兄的ae
        super(AutoEncoder_Miao, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        # 16 32 32
        self.conv1 = nn.Sequential(nn.Conv2d(
            in_channels=3,  # input height
            out_channels=16,
            kernel_size=3,  # filter size
            stride=1,  # filter movement/step
            padding=1), nn.LeakyReLU())
        # 32 16 16
        self.conv2 = nn.Sequential(nn.Conv2d(
            in_channels=16,  # input height
            out_channels=32,  # n_filters
            kernel_size=3,  # filter size
            stride=1,  # filter movement/step
            padding=1,
        ),
            nn.LeakyReLU(),  # activation
            nn.MaxPool2d(kernel_size=2))
        # 32 16 16
        self.conv3 = nn.Sequential(nn.Conv2d(
            in_channels=32,  # input height
            out_channels=32,  # n_filters
            kernel_size=5,  # filter size
            stride=1,  # filter movement/step
            padding=2,
        ),
            nn.LeakyReLU())  # activation)
        # 64 8 8
        self.conv4 = nn.Sequential(nn.Conv2d(
            in_channels=32,  # input height
            out_channels=64,  # n_filters
            kernel_size=5,  # filter size
            stride=1,  # filter movement/step
            padding=2,
        ),
            nn.LeakyReLU(),  # activation
            nn.MaxPool2d(kernel_size=2))
        # decoder
        # 128 8 8 -> 32 16 16
        self.ct0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=2, stride=2, padding=0), nn.LeakyReLU())
        # 64 8 8 -> 32 16 16
        self.ct1 = nn.Sequential(nn.ConvTranspose2d(
            in_channels=64,  # input height
            out_channels=32,  # n_filters
            kernel_size=2,  # filter size
            stride=2,  # filter movement/step
            padding=0,
        ),
            nn.LeakyReLU())
        #
        self.ct2 = nn.Sequential(nn.Conv2d(
            in_channels=32,  # input height
            out_channels=32,  # n_filters
            kernel_size=5,  # filter size
            stride=1,  # filter movement/step
            padding=2,
        ),
            nn.LeakyReLU())
        # -> 16 16 16
        self.ct3 = nn.Sequential(nn.ConvTranspose2d(
            in_channels=32,  # input height
            out_channels=16,  # n_filters
            kernel_size=5,  # filter size
            stride=1,  # filter movement/step
            padding=2,
        ))
        self.ct4 = nn.Sequential(nn.Conv2d(
            in_channels=16,  # input height
            out_channels=16,  # n_filters
            kernel_size=5,  # filter size
            stride=1,  # filter movement/step
            padding=2,
        ),
            nn.LeakyReLU())
        # ->16 32 32
        self.ct5 = nn.Sequential(nn.ConvTranspose2d(
            in_channels=16,  # input height
            out_channels=16,  # n_filters
            kernel_size=2,  # filter size
            stride=2,  # filter movement/step
            padding=0,
        ),
            nn.LeakyReLU())
        self.ct6 = nn.Sequential(nn.Conv2d(
            in_channels=16,  # input height
            out_channels=16,  # n_filters
            kernel_size=3,  # filter size
            stride=1,  # filter movement/step
            padding=1,
        ),
            nn.LeakyReLU())
        # -> 3 32 32
        self.ct7 = nn.Sequential(nn.ConvTranspose2d(
            in_channels=16,  # input height
            out_channels=3,  # n_filters
            kernel_size=5,  # filter size
            stride=1,  # filter movement/step
            padding=2,
        ),
            nn.LeakyReLU())
        self.ct8 = nn.Sequential(nn.Conv2d(
            in_channels=3,  # input height
            out_channels=3,  # n_filters
            kernel_size=3,  # filter size
            stride=1,  # filter movement/step
            padding=1,
        ))
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    '''
    训练的效果不错,resnet18 epoch200时候acc=0.956,loss=0.1652
    crossentropyloss自带softmax所以模型内部不用接softmax
    '''

    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet18_sigmoid(set_sigmoid=True):
    '''
    得到ResNet18_sigmoid
    :return:
    '''
    return ResNet_sigmoid(BasicBlock, [2, 2, 2, 2], set_sigmoid=set_sigmoid)


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def getResNet(name):
    '''
    :param name: 想要的resnet版本183450 101 152
    :return: 返回相应的resnet

    '''
    if "18" in name:
        return ResNet18()
    elif "34" in name:
        return ResNet34()
    elif "50" in name:
        return ResNet50()
    elif "101" in name:
        return ResNet101()
    elif "152" in name:
        return ResNet152()
    else:
        print(f"error,没有{name}模型")
        exit(1)


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())


class AutoEncoder_origin(nn.Module):
    '''
    原始
    cifar10 ae
    https://github.com/chenjie/PyTorch-CIFAR-10-autoencoder/blob/master/main.py
    效果如何不清楚
    '''

    def __init__(self):
        # 需要一个正常的重构损失,表示能将原始图像还原
        # 需要disc target两个标签的值要平均,一个loss
        # 两个loss相加
        # 还可以加一个chamferloss
        super(AutoEncoder_origin, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            # 			nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
            #             nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            #             nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
            #             nn.ReLU(),
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),  # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class AutoEncoder(nn.Module):
    '''
    起初实验都是用的这个,现在老师说这个结构不好,我试试苗师兄的
    自己写的,可以将两张图像何在一起
    目前是相邻两张图像合在一起,即随机何在一起,并没有判断两张图像target是否相同
    cifar10 ae
    https://github.com/chenjie/PyTorch-CIFAR-10-autoencoder/blob/master/main.py
    效果如何不清楚
    '''

    def __init__(self):
        # 需要一个正常的重构损失,表示能将原始图像还原
        # 需要disc target两个标签的值要平均,一个loss
        # 两个loss相加
        # 还可以加一个chamferloss
        super(AutoEncoder, self).__init__()
        # encoder Input size: [batch, 3, 32, 32]
        self.conv1 = nn.Sequential(nn.Conv2d(3, 12, 4, stride=2, padding=1), nn.BatchNorm2d(12),
                                   nn.ReLU())  # [batch, 12, 16, 16]
        self.conv2 = nn.Sequential(nn.Conv2d(12, 24, 4, stride=2, padding=1), nn.BatchNorm2d(24),
                                   nn.ReLU())  # [batch, 24, 8, 8]
        self.conv3 = nn.Sequential(nn.Conv2d(24, 48, 4, stride=2, padding=1), nn.BatchNorm2d(48),
                                   nn.ReLU())  # [batch, 48, 4, 4]
        '''nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]'''
        # decoder [batch, 48, 4, 4]->[batch, 3, 32, 32]
        self.tconv1 = nn.Sequential(nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1), nn.BatchNorm2d(24),
                                    nn.ReLU())  # [batch, 24, 8, 8]
        self.tconv2 = nn.Sequential(nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1), nn.BatchNorm2d(12),
                                    nn.ReLU())  # [batch, 12, 16, 16]
        self.tconv3 = nn.Sequential(nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1), nn.BatchNorm2d(3),
                                    nn.Sigmoid())  # [batch, 3, 32, 32]

        # 两张vector合在一起的[batch, 96, 4, 4] -> [batch, 24, 8, 8],代替tconv1
        self.tconv4 = nn.Sequential(nn.ConvTranspose2d(96, 24, 4, stride=2, padding=1),
                                    nn.ReLU())  # [batch, 24, 8, 8]

    def encoder(self, x):
        # x[batch, 3, 32, 32]
        x = self.conv1(x)  # [batch, 12, 16, 16]
        x = self.conv2(x)  # [batch, 24, 8, 8]
        x = self.conv3(x)  # [batch, 48, 4, 4]
        return x

    def decoder(self, z):
        # z[batch, 48, 4, 4]
        z = self.tconv1(z)  # [batch, 24, 8, 8]
        z = self.tconv2(z)  # [batch, 12, 16, 16]
        z = self.tconv3(z)  # [batch, 3, 32, 32]
        return z

    def decoder_virtual(self, double_z):
        # double_z [batch/2, 48*2, 4, 4]
        z = self.tconv4(double_z)  # [batch, 24, 8, 8]
        z = self.tconv2(z)  # [batch, 12, 16, 16]
        z = self.tconv3(z)  # [batch, 3, 32, 32]
        return z

    def generate_virtual(self, x):
        '''
        :param x:一个batch的图像样本,batch必须为整数
        :return: batch/2 个生成的虚假图像
        # 不需要用torch.random.shuffle()因为,我们这样相邻两个合在一起就已经相当于随机拼接了,何况dataloader也有shuffle
        # 不需要顾及z的batch不是偶数的情况,dataset是偶数,batchsize是偶数,所以batch一定是偶数
        '''
        z = self.encoder(x)  # [batch, 48, 4, 4]
        z = z.reshape(-1, z.shape[1] * 2, z.shape[2], z.shape[3])
        virtual = self.decoder_virtual(z)
        return virtual

    # 先不判断标签一致,写一个一个batch里随机混合的吧
    #
    def forward(self, x):
        '''

        :param x:
        :return: encodeddecoded
        '''
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AutoEncoder_origin_Miaoshixiong(nn.Module):
    '''
    原版苗师兄CODEs使用的ae
    改类名记得改super
    '''

    def __init__(self):
        super(AutoEncoder_origin_Miaoshixiong, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=3,  # input height
                out_channels=16,
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,
            ),
            nn.LeakyReLU(),  # activation
            nn.Conv2d(
                in_channels=16,  # input height
                out_channels=32,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,
            ),
            nn.LeakyReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=32,  # input height
                out_channels=32,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
            ),
            nn.LeakyReLU(),  # activation
            nn.Conv2d(
                in_channels=32,  # input height
                out_channels=64,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
            ),
            nn.LeakyReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64,  # input height
                out_channels=32,  # n_filters
                kernel_size=2,  # filter size
                stride=2,  # filter movement/step
                padding=0,
            ),
            nn.LeakyReLU(),  # activation
            nn.Conv2d(
                in_channels=32,  # input height
                out_channels=32,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
            ),
            nn.LeakyReLU(),  # activation
            nn.ConvTranspose2d(
                in_channels=32,  # input height
                out_channels=16,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
            ),
            nn.Conv2d(
                in_channels=16,  # input height
                out_channels=16,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
            ),
            nn.LeakyReLU(),  # activation
            nn.ConvTranspose2d(
                in_channels=16,  # input height
                out_channels=16,  # n_filters
                kernel_size=2,  # filter size
                stride=2,  # filter movement/step
                padding=0,
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=16,  # input height
                out_channels=16,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,
            ),
            nn.LeakyReLU(),  # activation
            nn.ConvTranspose2d(
                in_channels=16,  # input height
                out_channels=3,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=3,  # input height
                out_channels=3,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,
            ),
            # nn.ReLU(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AutoEncoder_Miao(nn.Module):
    '''
    改版自苗师兄的ae,不出意外,以后用这个

    注意:他这个没有加batchnorm,我不知道我要不要加上,但是不加的话,输出范围Wie[-0.1,1.1]超过[0,1]了
    '''

    def __init__(self):
        super(AutoEncoder_Miao, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        # 16 32 32
        self.conv1 = nn.Sequential(nn.Conv2d(
            in_channels=3,  # input height
            out_channels=16,
            kernel_size=3,  # filter size
            stride=1,  # filter movement/step
            padding=1), nn.LeakyReLU())
        # 32 16 16
        self.conv2 = nn.Sequential(nn.Conv2d(
            in_channels=16,  # input height
            out_channels=32,  # n_filters
            kernel_size=3,  # filter size
            stride=1,  # filter movement/step
            padding=1,
        ),
            nn.LeakyReLU(),  # activation
            nn.MaxPool2d(kernel_size=2))
        # 32 16 16
        self.conv3 = nn.Sequential(nn.Conv2d(
            in_channels=32,  # input height
            out_channels=32,  # n_filters
            kernel_size=5,  # filter size
            stride=1,  # filter movement/step
            padding=2,
        ),
            nn.LeakyReLU())  # activation)
        # 64 8 8
        self.conv4 = nn.Sequential(nn.Conv2d(
            in_channels=32,  # input height
            out_channels=64,  # n_filters
            kernel_size=5,  # filter size
            stride=1,  # filter movement/step
            padding=2,
        ),
            nn.LeakyReLU(),  # activation
            nn.MaxPool2d(kernel_size=2))
        # decoder
        # 128 8 8 -> 32 16 16
        self.ct0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=2, stride=2, padding=0), nn.LeakyReLU())
        # 64 8 8 -> 32 16 16
        self.ct1 = nn.Sequential(nn.ConvTranspose2d(
            in_channels=64,  # input height
            out_channels=32,  # n_filters
            kernel_size=2,  # filter size
            stride=2,  # filter movement/step
            padding=0,
        ),
            nn.LeakyReLU())
        #
        self.ct2 = nn.Sequential(nn.Conv2d(
            in_channels=32,  # input height
            out_channels=32,  # n_filters
            kernel_size=5,  # filter size
            stride=1,  # filter movement/step
            padding=2,
        ),
            nn.LeakyReLU())
        # -> 16 16 16
        self.ct3 = nn.Sequential(nn.ConvTranspose2d(
            in_channels=32,  # input height
            out_channels=16,  # n_filters
            kernel_size=5,  # filter size
            stride=1,  # filter movement/step
            padding=2,
        ))
        self.ct4 = nn.Sequential(nn.Conv2d(
            in_channels=16,  # input height
            out_channels=16,  # n_filters
            kernel_size=5,  # filter size
            stride=1,  # filter movement/step
            padding=2,
        ),
            nn.LeakyReLU())
        # ->16 32 32
        self.ct5 = nn.Sequential(nn.ConvTranspose2d(
            in_channels=16,  # input height
            out_channels=16,  # n_filters
            kernel_size=2,  # filter size
            stride=2,  # filter movement/step
            padding=0,
        ),
            nn.LeakyReLU())
        self.ct6 = nn.Sequential(nn.Conv2d(
            in_channels=16,  # input height
            out_channels=16,  # n_filters
            kernel_size=3,  # filter size
            stride=1,  # filter movement/step
            padding=1,
        ),
            nn.LeakyReLU())
        # -> 3 32 32
        self.ct7 = nn.Sequential(nn.ConvTranspose2d(
            in_channels=16,  # input height
            out_channels=3,  # n_filters
            kernel_size=5,  # filter size
            stride=1,  # filter movement/step
            padding=2,
        ),
            nn.LeakyReLU())
        self.ct8 = nn.Sequential(nn.Conv2d(
            in_channels=3,  # input height
            out_channels=3,  # n_filters
            kernel_size=3,  # filter size
            stride=1,  # filter movement/step
            padding=1,
        ), nn.Sigmoid())

    def encoder(self, x):
        x = self.conv1(x)  # 16,32,32
        x = self.conv2(x)  # 32,16,16
        x = self.conv3(x)  # 32,16,16
        x = self.conv4(x)  # 64,8,8
        return x

    def decoder(self, x):
        # 64,8,8
        x = self.ct1(x)  # 32,16,16
        x = self.ct2(x)
        x = self.ct3(x)  # 16,16,16
        x = self.ct4(x)
        x = self.ct5(x)  # 16,32,32
        x = self.ct6(x)
        x = self.ct7(x)  # 3,32,32
        x = self.ct8(x)
        return x

    def decoder_virtual(self, double_z):
        # double_z [batch/2, 64*2, 8, 8]
        x = self.ct0(double_z)  # 32 16 16
        x = self.ct2(x)
        x = self.ct3(x)  # 16,16,16
        x = self.ct4(x)
        x = self.ct5(x)  # 16,32,32
        x = self.ct6(x)
        x = self.ct7(x)  # 3,32,32
        x = self.ct8(x)
        return x

    def generate_virtual(self, data, label, set_encoded_detach, train_generate, num_classes=10, scale_generate=1):
        '''
        附带设置父母类别种类不一样
        生成虚假样本
        :param data:
        :param label:
        :param set_encoded_detach: encode之后detach一下,意思是不更新encoder的权重
        :param train_generate:是否为训练生成虚假样本,True则标签为[0.5,0.5,...] False则label都为0.1
        :param scale_generate: 虚假样本相对于正常样本的倍数,这个就锁死跟正常样本等量得了
        :param num_classes: 样本种类数
        :return: virtual_data,virtual_label
        '''
        scale_generate = int(scale_generate * 2)  # 需要多少个0.5倍
        batch_size = int(data.shape[0] * (scale_generate * 0.5))
        data = data.repeat_interleave(scale_generate + 1, dim=0)
        label = label.repeat_interleave(scale_generate + 1, dim=0)
        idx = torch.randperm(data.shape[0])
        data = data[idx].detach()
        label = label[idx].detach()
        # 虚假样本
        z = self.encoder(data)  # [batch, 64, 8, 8]
        z = z.reshape(-1, z.shape[1] * 2, z.shape[2], z.shape[3])
        if set_encoded_detach:
            z = z.detach()
        virtual_data = self.decoder_virtual(z)
        # 虚假标签
        virtual_label = F.one_hot(label, num_classes)
        index_0 = range(0, len(virtual_label), 2)
        index_1 = range(1, len(virtual_label), 2)
        virtual_label = virtual_label[index_0] + virtual_label[index_1]  # [0,1,1,0,...0]和[0,2,0,0...0]
        # ##排除相同类,就要在所有的索引中排除label有2的索引
        idx_2 = torch.where(virtual_label == 2)[0].cpu().numpy()  # label有2,也就是相同类生成的虚假样本
        idx_all = np.arange(0, virtual_label.shape[0])
        idx_1 = np.setdiff1d(idx_all, idx_2)  # 取补集,排除所有label有2的样本索引
        # ##按照这些索引取,同时只取batchsize张
        # virtual_data可不能detach,差点犯大错了
        virtual_data = virtual_data[idx_1][:batch_size]
        virtual_label = virtual_label[idx_1][:batch_size]
        # True为训练ae生成虚假样本,False为用ae生成的虚假样本进行压制训练
        if train_generate == True:
            # 训练生成虚假样本,我们虚假样本的标签设置为
            # virtual_label = virtual_label -0.1   # [0,0,1,1,0,0...,0] -> [-0.1,-0.1,0.9,0.9....,-0.1]
            virtual_label = virtual_label * 0.5  # [0,0,1,1,0,0...,0] -> [0 0 0.5,0.5,0,0]
            virtual_label = virtual_label.float()
        else:
            # 如果为压制训练,则virtual_laebl设置为都是0.1
            virtual_label = (torch.ones([len(virtual_data), num_classes]) * 0.1).cuda()

        virtual_label = virtual_label.detach()
        return virtual_data, virtual_label

    def generate_virtual_v1(self, data, label, set_encoded_detach, train_generate, num_classes=10):
        '''
        v1版本生成虚假样本,按照老师说的方法,利用两个len(data)长度所索引,也不用在这里限制父母类别不同
        应该在最后计算loss的时候将父母类别相同的loss设置为0就可以了,妙啊,老师太厉害了,我怎么就没想到呢
        嗯...然后实际还是有所改动,不然压制训练的时候,还是会存在父母类别相同的问题
        :return:
        '''
        index_0 = torch.randperm(len(data) * 2) % len(data)  # 多一倍的数量,因为label相同不要造成的损耗
        index_1 = torch.randperm(len(data) * 2) % len(data)
        index_notequal = label[index_0] != label[index_1]  # label不相同的索引
        index_0 = index_0[index_notequal][:len(data)]  # 要label不相同的,并且,弱水三千只取一瓢
        index_1 = index_1[index_notequal][:len(data)]  # 我太有才了
        # 到这里index_0 index_1就算做好了
        index_ji = range(1, len(data) * 2, 2)
        index_ou = range(0, len(data) * 2, 2)
        virtual_data = torch.zeros([data.shape[0] * 2, data.shape[1], data.shape[2], data.shape[3]]).cuda()
        virtual_data[index_ou] = data[index_0]
        virtual_data[index_ji] = data[index_1]
        virtual_data = virtual_data.detach()  # 原始数据到这里就配对好了
        label = F.one_hot(label, num_classes) * 0.5  # [0,0.5,0,...,0]
        virtual_label = label[index_0] + label[index_1]  # 都为[0,0.5,0.5,0,0]

        if train_generate == False:
            virtual_label = (torch.ones_like(virtual_label) * 0.1).detach().cuda()
        virtual_label = virtual_label.detach().cuda()

        # 生成虚假样本
        z = self.encoder(virtual_data)
        # print(f"z.shape={z.shape}")
        z = z.reshape(-1, z.shape[1] * 2, z.shape[2], z.shape[3])
        if set_encoded_detach:
            z = z.detach()
        virtual_data = self.decoder_virtual(z)
        # print(virtual_data.shape,label.shape)
        # print(label)
        return virtual_data, virtual_label

    def generate_virtual_v3(self, data, label, set_encoded_detach, train_generate, num_classes=10):
        '''
        v1版本生成虚假样本,按照老师说的方法,利用两个len(data)长度所索引,也不用在这里限制父母类别不同
        应该在最后计算loss的时候将父母类别相同的loss设置为0就可以了,妙啊,老师太厉害了,我怎么就没想到呢
        嗯...然后实际还是有所改动,不然压制训练的时候,还是会存在父母类别相同的问题,那时候还是改label啊笨蛋
        '''
        index_0 = torch.randperm(len(data))
        index_1 = torch.randperm(len(data))
        index_equal = label[index_0] == label[index_1]  # label相同的索引

        # 到这里index_0 index_1就算做好了
        index_ji = range(1, len(data) * 2, 2)
        index_ou = range(0, len(data) * 2, 2)
        virtual_data = torch.zeros([data.shape[0] * 2, data.shape[1], data.shape[2], data.shape[3]]).cuda()
        virtual_data[index_ou] = data[index_0]
        virtual_data[index_ji] = data[index_1]
        virtual_data = virtual_data.detach()  # 原始数据到这里就配对好了
        label = F.one_hot(label, num_classes) * 0.5  # [0,0.5,0,...,0]
        virtual_label = label[index_0] + label[index_1]  # 都为[0,0.5,0.5,0,0]
        virtual_label = torch.where(virtual_label == 1, 0, virtual_label)
        if train_generate == False:
            # 压制训练,相同类的loss设置成0
            virtual_label = (torch.ones_like(virtual_label) * 0.1).detach().cuda()
            virtual_label[index_equal] = torch.zeros([1, num_classes]).cuda()
        virtual_label = virtual_label.detach().cuda()

        # 生成虚假样本
        z = self.encoder(virtual_data)
        # print(f"z.shape={z.shape}")
        z = z.reshape(-1, z.shape[1] * 2, z.shape[2], z.shape[3])
        if set_encoded_detach:
            z = z.detach()
        virtual_data = self.decoder_virtual(z)
        # print(virtual_data.shape,label.shape)
        # print(label)
        return virtual_data, virtual_label

    def generate_virtual_v2(self, data, label, set_encoded_detach, train_generate, num_classes=10):
        '''
        v2版本生成虚假样本,两个z和label乘以对应权重再相加
        :return:
        '''
        index_0 = torch.randperm(len(data)).cuda()
        index_1 = torch.randperm(len(data)).cuda()
        if train_generate == True:
            weight_0 = torch.rand([label.shape[0], 1]).cuda()
        else:
            # 压制训练的时候,两个类别的权重应该为0.4到0.6之间
            weight_0 = torch.empty((label.shape[0], 1), dtype=torch.float32).uniform_(0.4, 0.6).cuda()
            index_notequal = label[index_0] != label[index_1]
            weight_0 = weight_0[index_notequal]
            index_0 = index_0[index_notequal]
            index_1 = index_1[index_notequal]
        label = F.one_hot(label, num_classes)  # [0,1,0,...,0]
        virtual_label = weight_0 * label[index_0] + (1 - weight_0) * label[index_1]

        # 处理数据
        z_0 = self.encoder(data[index_0])
        z_1 = self.encoder(data[index_1])
        if set_encoded_detach:
            z_0 = z_0.detach()
            z_1 = z_1.detach()

        weight_0 = weight_0.reshape(-1, 1, 1, 1)
        virtual_z = weight_0 * z_0 + (1 - weight_0) * z_1
        virtual_data = self.decoder(virtual_z)

        virtual_label = virtual_label.detach()
        # print(virtual_data.shape, virtual_label.shape)
        # print(virtual_data, virtual_label)
        return virtual_data, virtual_label

    def generate_virtual_by_add(self, x):
        pass

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AutoEncoder_Miao_containy(nn.Module):
    '''
    在苗师兄ae的基础上,借鉴FaderNetwork的思想,在decoder阶段加入y的标签
    '''

    def __init__(self, num_classes):
        super(AutoEncoder_Miao_containy, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.num_classes = num_classes
        # 16 32 32
        self.conv1 = nn.Sequential(nn.Conv2d(
            in_channels=3,  # input height
            out_channels=16,
            kernel_size=3,  # filter size
            stride=1,  # filter movement/step
            padding=1), nn.LeakyReLU())
        # 32 16 16
        self.conv2 = nn.Sequential(nn.Conv2d(
            in_channels=16,  # input height
            out_channels=32,  # n_filters
            kernel_size=3,  # filter size
            stride=1,  # filter movement/step
            padding=1,
        ),
            nn.LeakyReLU(),  # activation
            nn.MaxPool2d(kernel_size=2))
        # 32 16 16
        self.conv3 = nn.Sequential(nn.Conv2d(
            in_channels=32,  # input height
            out_channels=32,  # n_filters
            kernel_size=5,  # filter size
            stride=1,  # filter movement/step
            padding=2,
        ),
            nn.LeakyReLU())  # activation)
        # 64 8 8
        self.conv4 = nn.Sequential(nn.Conv2d(
            in_channels=32,  # input height
            out_channels=64,  # n_filters
            kernel_size=5,  # filter size
            stride=1,  # filter movement/step
            padding=2,
        ),
            nn.LeakyReLU(),  # activation
            nn.MaxPool2d(kernel_size=2))
        # decoder
        # 128 8 8 -> 32 16 16
        self.ct0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128 + self.num_classes, out_channels=32, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU())
        # 64 8 8 -> 32 16 16
        self.ct1 = nn.Sequential(nn.ConvTranspose2d(
            in_channels=64 + self.num_classes,  # input height
            out_channels=32,  # n_filters
            kernel_size=2,  # filter size
            stride=2,  # filter movement/step
            padding=0,
        ),
            nn.LeakyReLU())
        #
        self.ct2 = nn.Sequential(nn.Conv2d(
            in_channels=32 + self.num_classes,  # input height
            out_channels=32,  # n_filters
            kernel_size=5,  # filter size
            stride=1,  # filter movement/step
            padding=2,
        ),
            nn.LeakyReLU())
        # -> 16 16 16
        self.ct3 = nn.Sequential(nn.ConvTranspose2d(
            in_channels=32 + self.num_classes,  # input height
            out_channels=16,  # n_filters
            kernel_size=5,  # filter size
            stride=1,  # filter movement/step
            padding=2,
        ))
        self.ct4 = nn.Sequential(nn.Conv2d(
            in_channels=16 + self.num_classes,  # input height
            out_channels=16,  # n_filters
            kernel_size=5,  # filter size
            stride=1,  # filter movement/step
            padding=2,
        ),
            nn.LeakyReLU())
        # ->16 32 32
        self.ct5 = nn.Sequential(nn.ConvTranspose2d(
            in_channels=16 + self.num_classes,  # input height
            out_channels=16,  # n_filters
            kernel_size=2,  # filter size
            stride=2,  # filter movement/step
            padding=0,
        ),
            nn.LeakyReLU())
        self.ct6 = nn.Sequential(nn.Conv2d(
            in_channels=16 + self.num_classes,  # input height
            out_channels=16,  # n_filters
            kernel_size=3,  # filter size
            stride=1,  # filter movement/step
            padding=1,
        ),
            nn.LeakyReLU())
        # -> 3 32 32
        self.ct7 = nn.Sequential(nn.ConvTranspose2d(
            in_channels=16 + self.num_classes,  # input height
            out_channels=3,  # n_filters
            kernel_size=5,  # filter size
            stride=1,  # filter movement/step
            padding=2,
        ),
            nn.LeakyReLU())
        self.ct8 = nn.Sequential(nn.Conv2d(
            in_channels=3 + self.num_classes,  # input height
            out_channels=3,  # n_filters
            kernel_size=3,  # filter size
            stride=1,  # filter movement/step
            padding=1,
        ), nn.Sigmoid())

    def encoder(self, x):
        x = self.conv1(x)  # 16,32,32
        x = self.conv2(x)  # 32,16,16
        x = self.conv3(x)  # 32,16,16
        x = self.conv4(x)  # 64,8,8
        return x

    def decoder(self, x, y):
        # 64,8,8 -> 3 28 28
        if len(y.shape) == 1:
            y = F.one_hot(y, self.num_classes)
            print(y)
        assert y.shape == (x.shape[0], self.num_classes)
        y = y.unsqueeze(2).unsqueeze(3)  # [n, num_classes, 1, 1]
        for ct in [self.ct1, self.ct2, self.ct3, self.ct4, self.ct5, self.ct6, self.ct7, self.ct8]:
            x = torch.concat([x, y.expand(-1, -1, x.shape[2], x.shape[3])], dim=1)
            x = ct(x)
        return x

    def decoder_virtual(self, x, y):
        # 128,8,8 -> 3 28 28
        assert y.shape == (x.shape[0], self.num_classes)
        y = y.unsqueeze(2).unsqueeze(3)  # [n, num_classes, 1, 1]
        for ct in [self.ct0, self.ct2, self.ct3, self.ct4, self.ct5, self.ct6, self.ct7, self.ct8]:
            x = torch.concat([x, y.expand(-1, -1, x.shape[2], x.shape[3])], dim=1)
            x = ct(x)
        return x

    def forward(self, x, y):
        '''
        正常重构
        :param x:
        :param y:有无经过独热编码都可以
        :return:
        '''
        encoded = self.encoder(x)
        decoded = self.decoder(encoded, y)
        return decoded

    def generate_virtual(self, x, y, set_encode_detach=True, set_virtual_label_uniform=True):
        '''
        :param x:
        :param y:未经过独热编码之前的y
        :param set_encode_detach:
        :param set_virtual_label_uniform:
        :return:生成的虚假图像x和虚假label y
        '''
        x = self.encoder(x)
        if set_encode_detach:
            x = x.detach()

        idx_0, idx_1 = torch.randperm(len(x)).cuda(), torch.randperm(len(x)).cuda()
        idx_notequal = y[idx_0] != y[idx_1]
        idx_0, idx_1 = idx_0[idx_notequal], idx_1[idx_notequal]
        x = torch.concat([x[idx_0], x[idx_1]], dim=1)
        # virtual_label
        y_0 = F.one_hot(y[idx_0], self.num_classes).cuda()
        y_1 = F.one_hot(y[idx_1], self.num_classes).cuda()
        y = (y_0 + y_1) / 2.
        if set_virtual_label_uniform:
            y = (torch.ones_like(y) * 0.1).float().cuda()
        y = y.detach()
        # virtual_data
        x = self.decoder_virtual(x, y)
        return x, y


class ResNet_sigmoid(nn.Module):
    '''
    ResNet_sigmoid
    不是,训练的效果不错,resnet18 epoch200时候acc=0.956,loss=0.1652
    '''

    def __init__(self, block, num_blocks, num_classes=10, set_sigmoid=True):
        '''
        :param block:
        :param num_blocks:
        :param num_classes:
        :param set_sigmoid: 设置是否需要sigmoid
        '''
        super(ResNet_sigmoid, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, 1)
        self.set_sigmoid = set_sigmoid
        self.sigmoid = torch.nn.Sigmoid()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if self.set_sigmoid:
            out = self.sigmoid(out)
        return out


class Discriminator_WGAN_miao_cifar10(nn.Module):
    '''
    苗师兄wgan的discriminator,用于数据集 cifar10,cifar100,svhn
    这个没有接sigmoid,很是奇怪,mnist接sigmoid了 --- 实验发现,接了sigmoid要好一些.
    --实验结果: 效果不好,猜测是discriminator太强了.
    '''

    def __init__(self, set_sigmoid=False, ngpu=1):
        '''
        :param set_sigmoid: 最后的输出是否接sigmoid
        '''
        super(Discriminator_WGAN_miao_cifar10, self).__init__()
        nc = 3
        ndf = 32
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64  32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32  16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8 4
            # nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 8),
            # nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),

        )
        self.set_sigmoid = set_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # if input.is_cuda and self.ngpu > 1:
        #     output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        # else:
        output = self.main(input)
        if self.set_sigmoid:
            output = self.sigmoid(output)
        # print('-----------------------------')
        # print(output.shape)
        output = output.mean(0)
        # return output.view(-1, 1).squeeze(1)
        return output.view(1)


class simple_ae(nn.Module):
    '''
    wgan调通了的生成器
    '''

    def __init__(self):
        latent_size = 64
        n_channel = 3
        n_g_feature = 64
        super(simple_ae, self).__init__()
        # encoder
        self.c1 = nn.Sequential(nn.Conv2d(3, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.c2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.c3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU())
        self.c4 = nn.Sequential(nn.Conv2d(256, 64, 4, 1, 0), nn.BatchNorm2d(64), nn.ReLU())

        # dncoder
        self.ct1 = nn.Sequential(nn.ConvTranspose2d(64, 4 * 64, kernel_size=4, bias=False),
                                 nn.BatchNorm2d(4 * 64),
                                 nn.ReLU())
        self.ct2 = nn.Sequential(nn.ConvTranspose2d(4 * 64, 2 * 64, kernel_size=4, stride=2, padding=1, bias=False),
                                 nn.BatchNorm2d(2 * 64),
                                 nn.ReLU())
        self.ct3 = nn.Sequential(nn.ConvTranspose2d(2 * 64, 64, kernel_size=4, stride=2, padding=1, bias=False),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU())
        self.ct4 = nn.Sequential(nn.ConvTranspose2d(64, n_channel, kernel_size=4, stride=2, padding=1),
                                 nn.Sigmoid())

    def encoder(self, x):
        x = self.c1(x)  # 64, 16, 16
        x = self.c2(x)  # 128, 8, 8
        x = self.c3(x)  # 256, 4, 4
        x = self.c4(x)  # 64, 1, 1
        return x

    def decoder(self, z):
        # z [b,latensize=64,1,1]
        z = self.ct1(z)  # 256 4 4
        z = self.ct2(z)  # 128, 8, 8
        z = self.ct3(z)  # 64, 16, 16
        z = self.ct4(z)  # 3, 32, 32
        return z

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class simple_discriminator(nn.Module):
    '''
    简单discriminator,源码没有接sigmoid,我尝试加sigmoid之后直接完蛋.
    '''

    def __init__(self):
        super(simple_discriminator, self).__init__()
        n_d_feature = 64
        n_channel = 3
        self.dis = nn.Sequential(
            nn.Conv2d(n_channel, n_d_feature, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(n_d_feature, 2 * n_d_feature, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 * n_d_feature),
            nn.LeakyReLU(0.2),

            nn.Conv2d(2 * n_d_feature, 4 * n_d_feature, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4 * n_d_feature),
            nn.LeakyReLU(0.2),

            nn.Conv2d(4 * n_d_feature, 1, kernel_size=4)
        )

    def forward(self, x):
        x = self.dis(x)
        return x


class Discriminator_FaderNet(nn.Module):
    '''
    借鉴Fadernet的思想,自己搭建一个同样用途的鉴别器
    输入:encoded
    结构:两层卷积两层全连接
    '''

    def __init__(self, in_channels, in_size):
        super(Discriminator_FaderNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 4, 2, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 4, 2, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
        )
        in_features = int(in_channels * in_size * in_size / 16)
        self.proj_layers = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.reshape(x.shape[0], -1)
        x = self.proj_layers(x)
        # x = self.sigmoid(x)#这个应该也不接sigmoid的吧
        return x


class Discriminator_FaderNet(nn.Module):
    '''
    仿照FaderNet的结构
    用了三层421卷积和2层全连接
    '''

    def __init__(self, in_channel, in_size, num_classes):
        super(Discriminator_FaderNet, self).__init__()
        self.in_channel = in_channel
        self.in_size = in_size
        self.num_classes = num_classes
        self.c1 = nn.Sequential(
            nn.Conv2d(self.in_channel, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, True)
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True)
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True)
        )
        self.in_linear = int(512 * in_size / 8)
        self.proj = nn.Sequential(
            nn.Linear(self.in_linear, 256),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, self.num_classes)
        )

    def forward(self, z):
        z = self.c1(z)
        z = self.c2(z)
        z = self.c3(z)
        z = z.squeeze()
        z = self.proj(z)
        return z


def ex_ae_fadernet():
    pass
    num_classes = 10
    model = AutoEncoder_Miao_containy(num_classes=10).cuda()
    x = torch.rand([16, 3, 28, 28]).cuda()
    y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]).cuda()
    model.generate_virtual(x, y)


def ex_dis():
    model = discriminator_FaderNet(512, 4)
    x = torch.rand([4, 512, 4, 4])
    y = model(x)
    print(y)


if __name__ == "__main__":
    # test()
    # getResNet("1")
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # model = simple_ae().cuda()
    # for batch in range(1):
    #     x = torch.randn([16, 3, 32, 32]).cuda()
    #     pred = model.encoder(x)
    #     print(pred.shape)
    #     decoded = model.decoder(pred)
    #     print(decoded.shape)
    '''model=simple_discriminator()
    x=torch.randn([32,3,32,32])
    y=model(x).reshape(-1)
    print(y.shape)'''
    '''model = AutoEncoder_Miao()
    data = torch.tensor([0, 1, 2, 3, 0]).reshape([5, 1, 1, 1]).float()
    label = torch.tensor([0, 1, 2, 3, 0])
    model.generate_virtual_v1(data, label, True, 4)'''
    # ex_ae_fadernet()
    # ex_dis()
    # model_g = AutoEncoder_Miao_containy()
    model_g = Discriminator_FaderNet(64, 8, 10)
    # model_g=nn.Conv2d(3,16,3,1,1)
    x = torch.rand(4, 64, 8, 8)
    print(model_g)
    y = model_g(x)
    print(y.shape)
