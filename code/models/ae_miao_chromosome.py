'''
染色体相关实验的模型
'''
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AutoEncoder_Miao_crossover(nn.Module):
    '''
    其实就是苗师兄最早的ae,这里用作染色体相关的ae
    交叉使用随机生成的tensor向量,不科学系
    '''

    def __init__(self, num_classes):
        super(AutoEncoder_Miao_crossover, self).__init__()
        self.num_classes = num_classes
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
        self.ct0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU())
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
        x = self.ct7(x)  # 3,32, 32
        x = self.ct8(x)
        return x

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def crossover(self, z1, z2, k):
        # 交叉操作,交叉比例为1/k
        index_main = torch.where(torch.randint(0, k, [z1.shape[0], z1.shape[1]]) != 0, 1, 0).cuda()
        index_main = index_main.view([z1.shape[0], z1.shape[1], 1, 1])
        index_minor = 1 - index_main
        z1 = z1 * index_main + z2 * index_minor
        z2 = z2 * index_main + z1 * index_minor
        return z1, z2

    def decoder_virtual(self, z1, z2, scale):
        '''
        暂时不考虑相同类的问题
        :param encoded:
        :return:
        '''
        z1, z2 = self.crossover(z1, z2, scale)

        z1, z2 = self.ct1(z1), self.ct1(z2)  # 32,16,16
        z1, z2 = self.ct2(z1), self.ct2(z2)
        z1, z2 = self.crossover(z1, z2, scale)

        z1, z2 = self.ct3(z1), self.ct3(z2)  # 16,16,16
        z1, z2 = self.ct4(z1), self.ct4(z2)
        z1, z2 = self.crossover(z1, z2, scale)

        z1, z2 = self.ct5(z1), self.ct5(z2)  # 16,32,32
        z1, z2 = self.ct6(z1), self.ct6(z2)
        z1, z2 = self.crossover(z1, z2, scale)

        z1, z2 = self.ct7(z1), self.ct7(z2)  # 3,32,32
        z1, z2 = self.ct8(z1), self.ct8(z2)

        return z1, z2

    def generate_virtual(self, data, label, set_differentlabel, set_virtuallabel_uniform, scale,
                         set_test=False):
        '''
        附带设置父母类别种类不一样
        生成虚假样本
        :param data:
        :param label:
        :param set_differentlabel: 父母类别是否限制different
        :param set_virtuallabel_uniform: 压制训练时候设置均匀标签,true的时候virtual_data也会detach
        :return: virtual_data, virtual_label.detach()
        '''
        encoded = self.encoder(data)  # [batch, 64, 8, 8]
        # 索引
        index1 = torch.arange(0, len(encoded)).cuda()
        index2 = torch.randperm(len(encoded)).cuda()
        if set_differentlabel:
            # 删去相同类
            index_notequal = label[index1] != label[index2]
            index1 = index1[index_notequal]
            index2 = index2[index_notequal]

        # 虚假样本
        z1 = encoded[index1].detach()
        z2 = encoded[index2].detach()
        virtual_data1, virtual_data2 = self.decoder_virtual(z1, z2, scale=scale)
        virtual_data = torch.concat([virtual_data1, virtual_data2], dim=0)

        # 虚假标签
        virtual_label = F.one_hot(label, self.num_classes)
        virtual_label = (virtual_label[index1] + virtual_label[index2]) * 0.5
        virtual_label = torch.concat([virtual_label, virtual_label], dim=0)

        # True则设置virtual_label为均匀
        if set_virtuallabel_uniform:
            # 如果为压制训练,则返回的virtual_data应detach
            virtual_data = virtual_data.detach()
            # 如果为压制训练,则virtual_laebl设置为都是1/num_classes
            virtual_label = (torch.ones([len(virtual_data), self.num_classes]) / self.num_classes).cuda()
        if not set_test:
            return virtual_data, virtual_label, (index1, index2)
        else:
            return data[index1], data[index2], label[index1], label[index2], virtual_data1, virtual_data2, virtual_label

    def generate_one_virtual(self, data0, data1, scale):
        '''
        generate only one virtual data
        生成虚假样本
        :param data:
        :param label:
        :param set_differentlabel: 父母类别是否限制different
        :param set_virtuallabel_uniform: 压制训练时候设置均匀标签,true的时候virtual_data也会detach
        :return: virtual_data, virtual_label.detach()
        '''
        if len(data0.shape) == 3:
            data0 = torch.unsqueeze(data0, 0)
            data1 = torch.unsqueeze(data1, 0)
        encoded0, encoded1 = self.encoder(data0), self.encoder(data1)  # [batch, 64, 8, 8]
        # 索引
        # 虚假样本
        virtual_data0, virtual_data1 = self.decoder_virtual(encoded0, encoded1, scale=scale)

        # 虚假标签

        return virtual_data0, virtual_data1

    ##########
    def generate_virtual_anquanzhong(self, data, label, set_differentlabel, set_virtuallabel_uniform, scale,
                                     set_test=False):
        '''
        附带设置父母类别种类不一样
        生成虚假样本
        :param data:
        :param label:
        :param set_differentlabel: 父母类别是否限制different
        :param set_virtuallabel_uniform: 压制训练时候设置均匀标签,true的时候virtual_data也会detach
        :return: virtual_data, virtual_label.detach()
        '''
        weight_scale = 1 - 1 / scale
        encoded = self.encoder(data)  # [batch, 64, 8, 8]
        # 索引
        index1 = torch.arange(0, len(encoded)).cuda()
        index2 = torch.randperm(len(encoded)).cuda()
        if set_differentlabel:
            # 删去相同类
            index_notequal = label[index1] != label[index2]
            index1 = index1[index_notequal]
            index2 = index2[index_notequal]

        # 虚假样本
        z1 = encoded[index1].detach()
        z2 = encoded[index2].detach()
        virtual_data1, virtual_data2 = self.decoder_virtual(z1, z2, scale=scale)
        virtual_data = torch.concat([virtual_data1, virtual_data2], dim=0)

        # 虚假标签
        virtual_label = F.one_hot(label, self.num_classes)
        virtual_label = weight_scale * virtual_label[index1] + (1 / scale) * virtual_label[index2]
        virtual_label = torch.concat([virtual_label, virtual_label], dim=0)

        # True则设置virtual_label为均匀
        if set_virtuallabel_uniform:
            # 如果为压制训练,则返回的virtual_data应detach
            virtual_data = virtual_data.detach()
            # 如果为压制训练,则virtual_laebl设置为都是1/num_classes
            virtual_label = (torch.ones([len(virtual_data), self.num_classes]) / self.num_classes).cuda()
        if not set_test:
            return virtual_data, virtual_label
        else:
            return data[index1], data[index2], label[index1], label[index2], virtual_data1, virtual_data2, virtual_label


class AutoEncoder_Miao_crossover_learnebalecrossover(nn.Module):
    '''
    只交换1/8 1/16
    其实就是苗师兄最早的ae,这里用作染色体相关的ae
    交叉使用nn.Parameter
    '''

    def __init__(self, num_classes):
        super(AutoEncoder_Miao_crossover, self).__init__()
        self.num_classes = num_classes
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
        self.ct0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU())
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
        x = self.ct7(x)  # 3,32, 32
        x = self.ct8(x)
        return x

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def crossover(self, z1, z2, k):
        # 交叉操作,交叉比例为1/k
        index_main = torch.where(torch.randint(0, k, [z1.shape[0], z1.shape[1]]) != 0, 1, 0).cuda()
        index_main = index_main.view([z1.shape[0], z1.shape[1], 1, 1])
        index_minor = 1 - index_main
        z1 = z1 * index_main + z2 * index_minor
        z2 = z2 * index_main + z1 * index_minor
        return z1, z2

    def decoder_virtual(self, z1, z2, scale):
        '''
        暂时不考虑相同类的问题
        :param encoded:
        :return:
        '''
        z1, z2 = self.crossover(z1, z2, scale)

        z1, z2 = self.ct1(z1), self.ct1(z2)  # 32,16,16
        z1, z2 = self.ct2(z1), self.ct2(z2)
        z1, z2 = self.crossover(z1, z2, scale)

        z1, z2 = self.ct3(z1), self.ct3(z2)  # 16,16,16
        z1, z2 = self.ct4(z1), self.ct4(z2)
        z1, z2 = self.crossover(z1, z2, scale)

        z1, z2 = self.ct5(z1), self.ct5(z2)  # 16,32,32
        z1, z2 = self.ct6(z1), self.ct6(z2)
        z1, z2 = self.crossover(z1, z2, scale)

        z1, z2 = self.ct7(z1), self.ct7(z2)  # 3,32,32
        z1, z2 = self.ct8(z1), self.ct8(z2)

        return z1, z2

    def generate_virtual(self, data, label, set_differentlabel, set_virtuallabel_uniform, scale, set_test=False):
        '''
        附带设置父母类别种类不一样
        生成虚假样本
        :param data:
        :param label:
        :param set_differentlabel: 父母类别是否限制different
        :param set_virtuallabel_uniform: 压制训练时候设置均匀标签,true的时候virtual_data也会detach
        :return: virtual_data, virtual_label.detach()
        '''
        encoded = self.encoder(data)  # [batch, 64, 8, 8]
        # 索引
        index1 = torch.arange(0, len(encoded)).cuda()
        index2 = torch.randperm(len(encoded)).cuda()
        if set_differentlabel:
            # 删去相同类
            index_notequal = label[index1] != label[index2]
            index1 = index1[index_notequal]
            index2 = index2[index_notequal]

        # 虚假样本
        z1 = encoded[index1].detach()
        z2 = encoded[index2].detach()
        virtual_data1, virtual_data2 = self.decoder_virtual(z1, z2, scale=scale)
        virtual_data = torch.concat([virtual_data1, virtual_data2], dim=0)

        # 虚假标签
        virtual_label = F.one_hot(label, self.num_classes)
        virtual_label = (virtual_label[index1] + virtual_label[index2]) * 0.5
        virtual_label = torch.concat([virtual_label, virtual_label], dim=0)

        # True则设置virtual_label为均匀
        if set_virtuallabel_uniform:
            # 如果为压制训练,则返回的virtual_data应detach
            virtual_data = virtual_data.detach()
            # 如果为压制训练,则virtual_laebl设置为都是1/num_classes
            virtual_label = (torch.ones([len(virtual_data), self.num_classes]) / self.num_classes).cuda()

        virtual_label = virtual_label.detach()
        if not set_test:
            return virtual_data, virtual_label
        else:
            return data[index1], data[index2], label[index1], label[index2], virtual_data1, virtual_data2, virtual_label

    def generate_one_virtual(self, data0, data1, scale):
        '''
        generate only one virtual data
        生成虚假样本
        :param data:
        :param label:
        :param set_differentlabel: 父母类别是否限制different
        :param set_virtuallabel_uniform: 压制训练时候设置均匀标签,true的时候virtual_data也会detach
        :return: virtual_data, virtual_label.detach()
        '''
        if len(data0.shape) == 3:
            data0 = torch.unsqueeze(data0, 0)
            data1 = torch.unsqueeze(data1, 0)
        encoded0, encoded1 = self.encoder(data0), self.encoder(data1)  # [batch, 64, 8, 8]
        # 索引
        # 虚假样本
        virtual_data0, virtual_data1 = self.decoder_virtual(encoded0, encoded1, scale=scale)

        # 虚假标签

        return virtual_data0, virtual_data1


class AutoEncoder_Miao_crossover_tangloss(nn.Module):
    '''
    其实就是苗师兄最早的ae,这里用作染色体相关的ae
    只交换1/8 1/16
    使用tangloss,那就得返回原始label标签
    '''

    def __init__(self, num_classes):
        super(AutoEncoder_Miao_crossover_tangloss, self).__init__()
        self.num_classes = num_classes
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
        self.ct0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU())
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
        x = self.ct7(x)  # 3,32, 32
        x = self.ct8(x)
        return x

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def crossover(self, z1, z2, k):
        # 交叉操作,交叉比例为1/k
        index_main = torch.where(torch.randint(0, k, [z1.shape[0], z1.shape[1]]) != 0, 1, 0).cuda()
        index_main = index_main.view([z1.shape[0], z1.shape[1], 1, 1])
        index_minor = 1 - index_main
        z1 = z1 * index_main + z2 * index_minor
        z2 = z2 * index_main + z1 * index_minor
        return z1, z2

    def decoder_virtual(self, z1, z2, scale):
        '''
        暂时不考虑相同类的问题
        :param encoded:
        :return:
        '''
        z1, z2 = self.crossover(z2, z1, scale)

        z1, z2 = self.ct1(z1), self.ct1(z2)  # 32,16,16
        z1, z2 = self.ct2(z1), self.ct2(z2)
        z1, z2 = self.crossover(z1, z2, scale)

        z1, z2 = self.ct3(z1), self.ct3(z2)  # 16,16,16
        z1, z2 = self.ct4(z1), self.ct4(z2)
        z1, z2 = self.crossover(z1, z2, scale)

        z1, z2 = self.ct5(z1), self.ct5(z2)  # 16,32,32
        z1, z2 = self.ct6(z1), self.ct6(z2)
        z1, z2 = self.crossover(z1, z2, scale)

        z1, z2 = self.ct7(z1), self.ct7(z2)  # 3,32,32
        z1, z2 = self.ct8(z1), self.ct8(z2)

        return z1, z2

    def generate_virtual(self, data, label, set_differentlabel, set_originlabel, set_virtuallabel_uniform, scale,
                         set_test=False):
        '''
        附带设置父母类别种类不一样
        生成虚假样本
        :param data:
        :param label:
        :param set_differentlabel: 父母类别是否限制different
        :param set_virtuallabel_uniform: 压制训练时候设置均匀标签,true的时候virtual_data也会detach
        :param set_originlabel: 返回的是图像原始的标签
        :return: virtual_data, virtual_label.detach()
        '''
        encoded = self.encoder(data)  # [batch, 64, 8, 8]
        # 索引
        index1 = torch.arange(0, len(encoded)).cuda()
        index2 = torch.randperm(len(encoded)).cuda()
        if set_differentlabel:
            # 删去相同类
            index_notequal = label[index1] != label[index2]
            index1 = index1[index_notequal]
            index2 = index2[index_notequal]

        # 虚假样本
        z1 = encoded[index1].detach()
        z2 = encoded[index2].detach()
        virtual_data1, virtual_data2 = self.decoder_virtual(z1, z2, scale=scale)
        virtual_data = torch.concat([virtual_data1, virtual_data2], dim=0)

        # 虚假标签
        virtual_label = F.one_hot(label, self.num_classes)
        virtual_label = (virtual_label[index1] + virtual_label[index2]) * 0.5
        virtual_label = torch.concat([virtual_label, virtual_label], dim=0)

        # True则设置virtual_label为均匀
        if set_virtuallabel_uniform:
            # 如果为压制训练,则返回的virtual_data应detach
            virtual_data = virtual_data.detach()
            # 如果为压制训练,则virtual_laebl设置为都是1/num_classes
            virtual_label = (torch.ones([len(virtual_data), self.num_classes]) / self.num_classes).cuda()
        elif set_originlabel:
            virtual_label = torch.concat([label[index1], label[index2]], dim=0)
        virtual_label = virtual_label.detach()

        if not set_test:
            return virtual_data, virtual_label
        else:
            return data[index1], data[index2], label[index1], label[index2], virtual_data1, virtual_data2, virtual_label

    def generate_one_virtual(self, data0, data1, scale):
        '''
        generate only one virtual data
        生成虚假样本
        :param data:
        :param label:
        :param set_differentlabel: 父母类别是否限制different
        :param set_virtuallabel_uniform: 压制训练时候设置均匀标签,true的时候virtual_data也会detach
        :return: virtual_data, virtual_label.detach()
        '''
        if len(data0.shape) == 3:
            data0 = torch.unsqueeze(data0, 0)
            data1 = torch.unsqueeze(data1, 0)
        encoded0, encoded1 = self.encoder(data0), self.encoder(data1)  # [batch, 64, 8, 8]
        # 索引
        # 虚假样本
        virtual_data0, virtual_data1 = self.decoder_virtual(encoded0, encoded1, scale=scale)

        # 虚假标签

        return virtual_data0, virtual_data1


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "cuda:0"
    model = AutoEncoder_Miao_crossover_tangloss(num_classes=10).cuda()
    num = 6
    x = torch.randn([num, 3, 32, 32]).cuda()
    y = model(x)
    print(y.shape)
    y = torch.randint(0, 9, [num])
    print(y)
    x, y = model.generate_virtual(x, y, set_differentlabel=True, set_virtuallabel_uniform=False, set_originlabel=True,
                                  scale=8)
    print(x.shape, y.shape)
    print(y)
