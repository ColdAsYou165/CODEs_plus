import torch.nn as nn
# import torch.utils.model_zoo as model_zoo


# from senet.bs_module import BSLayer, BSLayerLargerConv, BSLayer_NoBNReLU, BSLayerAfterReLU2, BSLayerFcRelu_NoBNReLU, \
#     BSLayerWithVar, BSLayerWithVar_LargerConv, BSLayerWithVar_LargerConv_And_CertainOutChannel, \
#     BSLayerWithVar_LargerConv_And_CertainOutChannel_AnotherRemap


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# class ImageNetBSBasicBlockWithVar_LargerConv_StartfromBegin_AnotherRemap(nn.Module):
#     expansion = 1
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
#                  norm_layer=None, reduction=16, numberOfClass=10, div=2):
#         super(ImageNetBSBasicBlockWithVar_LargerConv_StartfromBegin_AnotherRemap, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#
#         self.inplane = inplanes
#         self.outplane = planes
#         self.bs = BSLayerWithVar_LargerConv_And_CertainOutChannel_AnotherRemap(inplanes, planes, stride, reduction,
#                                                                                numberOfClass, div)
#         if inplanes != planes:
#             self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
#                                             nn.BatchNorm2d(planes))
#         else:
#             self.downsample = lambda x: x
#         self.stride = stride
#
#     def forward(self, x):
#         residual = self.downsample(x)
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         out += residual
#         dec = self.bs(x, out)
#         out = self.relu(dec)
#
#         return out
#
#     def forward_withVar(self, x, y):
#
#         residual = self.downsample(x)
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         out += residual
#         dec, var = self.bs.forward_withVar(x, out, y)
#         out = self.relu(dec)
#
#         return out, var
#
#
# class ImageNetBSResNetWithVar(nn.Module):
#     def __init__(self, block, layers, num_classes=1000, reduction=16, div=2, zero_init_residual=False,
#                  groups=1, width_per_group=64, replace_stride_with_dilation=None,
#                  norm_layer=None):
#         super(ImageNetBSResNetWithVar, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         self._norm_layer = norm_layer
#
#         self.inplanes = 64
#         self.dilation = 1
#         if replace_stride_with_dilation is None:
#             # each element in the tuple indicates if we should replace
#             # the 2x2 stride with a dilated convolution instead
#             replace_stride_with_dilation = [False, False, False]
#         if len(replace_stride_with_dilation) != 3:
#             raise ValueError("replace_stride_with_dilation should be None "
#                              "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
#         self.groups = groups
#         self.base_width = width_per_group
#         self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#         self.bn1 = norm_layer(self.inplanes)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer_withList(block, 64, layers[0], reduction=reduction, classnum=num_classes,
#                                                 div=div)
#         self.layer2 = self._make_layer_withList(block, 128, layers[1], stride=2, reduction=reduction,
#                                                 classnum=num_classes, div=div)
#         self.layer3 = self._make_layer_withList(block, 256, layers[2], stride=2, reduction=reduction,
#                                                 classnum=num_classes, div=div)
#         self.layer4 = self._make_layer_withList(block, 512, layers[3], stride=2, reduction=reduction,
#                                                 classnum=num_classes, div=div)
#
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512 * block.expansion, num_classes)
#
#         self.initialize()
#
#     def initialize(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
#         norm_layer = self._norm_layer
#         downsample = None
#         previous_dilation = self.dilation
#         if dilate:
#             self.dilation *= stride
#             stride = 1
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 conv1x1(self.inplanes, planes * block.expansion, stride),
#                 norm_layer(planes * block.expansion),
#             )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
#                             self.base_width, previous_dilation, norm_layer))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes, groups=self.groups,
#                                 base_width=self.base_width, dilation=self.dilation,
#                                 norm_layer=norm_layer))
#
#         return nn.Sequential(*layers)
#
#     def _make_layer_withList(self, block, planes, blocks, reduction, classnum, div, stride=1, dilate=False):
#         norm_layer = self._norm_layer
#         downsample = None
#         previous_dilation = self.dilation
#         if dilate:
#             self.dilation *= stride
#             stride = 1
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 conv1x1(self.inplanes, planes * block.expansion, stride),
#                 norm_layer(planes * block.expansion),
#             )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
#                             self.base_width, previous_dilation, norm_layer, reduction=reduction, numberOfClass=classnum,
#                             div=div))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes, groups=self.groups,
#                                 base_width=self.base_width, dilation=self.dilation,
#                                 norm_layer=norm_layer, reduction=reduction, numberOfClass=classnum, div=div))
#
#         return nn.ModuleList(layers)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         for i, layer in enumerate(self.layer1):
#             x = layer(x)
#         for i, layer in enumerate(self.layer2):
#             x = layer(x)
#         for i, layer in enumerate(self.layer3):
#             x = layer(x)
#         for i, layer in enumerate(self.layer4):
#             x = layer(x)
#
#         x = self.avgpool(x)
#         x = x.reshape(x.size(0), -1)
#         x = self.fc(x)
#
#         return x
#
#     def forwardByTrain(self, x, y):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         var = 0
#         num = 0.0
#
#         # print("before layer1 ")
#         # print(x.shape)
#         for i, layer in enumerate(self.layer1):
#             x, var1 = layer.forward_withVar(x, y)
#             var += var1
#             num += 1.0
#         # print("after layer1 ")
#         # print(x.shape)
#         for i, layer in enumerate(self.layer2):
#             x, var2 = layer.forward_withVar(x, y)
#             var += var2
#             num += 1.0
#
#         for i, layer in enumerate(self.layer3):
#             x, var3 = layer.forward_withVar(x, y)
#             var += var3
#             num += 1.0
#
#         for i, layer in enumerate(self.layer4):
#             x, var3 = layer.forward_withVar(x, y)
#             var += var3
#             num += 1.0
#
#         x = self.avgpool(x)
#         x = x.reshape(x.size(0), -1)
#         x = self.fc(x)
#
#         return x, var / num


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


class ImageNetBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, adv=False):
        super(ImageNetBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = Mixturenorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, adv)
        # print("stride=", stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, adv)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, adv=False):
        # print("block", adv)
        identity = x

        out = self.conv1(x)
        out = self.bn1(out, adv)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, adv)

        if self.downsample is not None:
            m = self.downsample[0](x)
            identity = self.downsample[1](m, adv)

        out += identity
        out = self.relu(out)

        return out


class ImageNetResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ImageNetResNet, self).__init__()
        if norm_layer is None:
            norm_layer = Mixturenorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = Mixturenorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, adv=False):
        norm_layer = self._norm_layer

        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.ModuleList([
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion, adv)]
            )
        else:
            downsample = None

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, adv))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, adv=adv))

        return nn.ModuleList(layers)

    def forward(self, x, adv=False):
        x = self.conv1(x)
        x = self.bn1(x, adv)
        x = self.relu(x)
        x = self.maxpool(x)
        for layer in self.layer1:
            x = layer(x, adv)
        for layer in self.layer2:
            x = layer(x, adv)
        for layer in self.layer3:
            x = layer(x, adv)
        for layer in self.layer4:
            x = layer(x, adv)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


def imagetNet_resnet18(num_classes, **kwargs):  ####################
    model = ImageNetResNet(ImageNetBasicBlock, [2, 2, 2, 2], num_classes, **kwargs)
    return model


# def imagetNet_bs_resnet18_withVar_LargerConv_fromBegin_AnotherRemap(num_classes, **kwargs):  ####################
#     model = ImageNetBSResNetWithVar(ImageNetBSBasicBlockWithVar_LargerConv_StartfromBegin_AnotherRemap, [2, 2, 2, 2],
#                                     num_classes, **kwargs)
#     return model
#
#
# def imagetNet_bs_resnet56_withVar_LargerConv_fromBegin_AnotherRemap(num_classes, **kwargs):
#     model = ImageNetBSResNetWithVar(ImageNetBSBasicBlockWithVar_LargerConv_StartfromBegin_AnotherRemap, 9, num_classes,
#                                     **kwargs)
#     return model
#
#
# ###########################################################


class BottleneckImageNet(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, adv=False):
        super(BottleneckImageNet, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = Mixturenorm2d(planes, adv)
        # print("stride=",stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = Mixturenorm2d(planes, adv)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = Mixturenorm2d(planes * self.expansion, adv)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, adv=False):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out, adv)
        out = self.relu(out)

        out = self.conv2(out)
        # print('self.stride:',self.stride)
        out = self.bn2(out, adv)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out, adv)

        if self.downsample is not None:
            m = self.downsample[0](x)
            residual = self.downsample[1](m, adv)

        out += residual
        out = self.relu(out)

        return out


class ResNetImageNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNetImageNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = Mixturenorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, adv=False):

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.ModuleList([
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                Mixturenorm2d(planes * block.expansion, adv)]
            )
        else:
            downsample = None

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, adv))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, adv=False))

        return nn.ModuleList(layers)

    def forward(self, x, adv=False):
        x = self.conv1(x)
        x = self.bn1(x, adv)
        x = self.relu(x)
        x = self.maxpool(x)
        for layer in self.layer1:
            x = layer(x, adv=adv)
        for layer in self.layer2:
            x = layer(x, adv=adv)
        for layer in self.layer3:
            x = layer(x, adv=adv)
        for layer in self.layer4:
            x = layer(x, adv=adv)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# class BasicBlockImageNet(nn.Module):
#     expansion = 1
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(BasicBlockImageNet, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
#
#
# from senet.bs_module import BSLayerWithVar_LargerConv_Conditional
# import torch
# from torch.autograd import Variable
#
#
# class BottleneckImageNet_ResDec(nn.Module):
#     expansion = 4
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16, numofclass=10, div=2, binary=False):
#         super(BottleneckImageNet_ResDec, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes + div, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes + div, planes, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes + div, planes * self.expansion, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#
#         self.div = div
#         self.bs = BSLayerWithVar_LargerConv_Conditional(inplanes, planes, stride=stride, reduction=reduction,
#                                                         numofclass=numofclass, div=div, binary=binary)
#
#     def forward(self, x):
#         residual = x
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         dec, var, entropy, diversity = self.bs(x)
#         dec = dec.contiguous()
#
#         b, c, h, w = x.size()
#         dec1 = dec.view(b, self.div, 1, 1)
#         dec1 = dec1.expand(b, self.div, h, w)
#         z = [x, dec1]
#         x = torch.cat(z, 1)
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         x = self.relu(out)
#
#         b, c, h, w = x.size()
#         dec2 = dec.view(b, self.div, 1, 1)
#         dec2 = dec2.expand(b, self.div, h, w)
#         z = [x, dec2]
#         x = torch.cat(z, 1)
#
#         out = self.conv2(x)
#         out = self.bn2(out)
#         x = self.relu(out)
#
#         b, c, h, w = x.size()
#         dec3 = dec.view(b, self.div, 1, 1)
#         dec3 = dec3.expand(b, self.div, h, w)
#         z = [x, dec3]
#         x = torch.cat(z, 1)
#
#         out = self.conv3(x)
#         out = self.bn3(out)
#         out += residual
#         out = self.relu(out)
#
#         return out
#
#     def forward_withVar(self, x, y):
#         dec, var, entropy, diversity = self.bs.forward_withVar(x, y)
#
#         dec = dec.contiguous()
#
#         residual = x
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         b, c, h, w = x.size()
#         dec1 = dec.view(b, self.div, 1, 1)
#         dec1 = dec1.expand(b, self.div, h, w)
#
#         z = [x, dec1]
#         x = torch.cat(z, 1)
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         x = self.relu(out)
#
#         b, c, h, w = x.size()
#         dec2 = dec.view(b, self.div, 1, 1)
#         dec2 = dec2.expand(b, self.div, h, w)
#
#         z = [x, dec2]
#         x = torch.cat(z, 1)
#
#         out = self.conv2(x)
#         out = self.bn2(out)
#         x = self.relu(out)
#
#         b, c, h, w = x.size()
#         dec3 = dec.view(b, self.div, 1, 1)
#         dec3 = dec3.expand(b, self.div, h, w)
#
#         z = [x, dec3]
#         x = torch.cat(z, 1)
#
#         out = self.conv3(x)
#         out = self.bn3(out)
#         out += residual
#         out = self.relu(out)
#
#         return out, var, entropy, diversity
#
#
# class BottleneckImageNet_ResDec_WithRepul(nn.Module):
#     expansion = 4
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16, numofclass=10, div=2, binary=False):
#         super(BottleneckImageNet_ResDec_WithRepul, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes + div, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes + div, planes, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes + div, planes * self.expansion, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#
#         self.div = div
#         self.bs = BSLayerWithVar_LargerConv_Conditional(inplanes, planes, stride=stride, reduction=reduction,
#                                                         numofclass=numofclass, div=div, binary=binary)
#
#     def forward(self, x):
#         residual = x
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         dec = self.bs(x)
#         dec = dec.contiguous()
#
#         b, c, h, w = x.size()
#         dec1 = dec.view(b, self.div, 1, 1)
#         dec1 = dec1.expand(b, self.div, h, w)
#         z = [x, dec1]
#         x = torch.cat(z, 1)
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         x = self.relu(out)
#
#         b, c, h, w = x.size()
#         dec2 = dec.view(b, self.div, 1, 1)
#         dec2 = dec2.expand(b, self.div, h, w)
#         z = [x, dec2]
#         x = torch.cat(z, 1)
#
#         out = self.conv2(x)
#         out = self.bn2(out)
#         x = self.relu(out)
#
#         b, c, h, w = x.size()
#         dec3 = dec.view(b, self.div, 1, 1)
#         dec3 = dec3.expand(b, self.div, h, w)
#         z = [x, dec3]
#         x = torch.cat(z, 1)
#
#         out = self.conv3(x)
#         out = self.bn3(out)
#         out += residual
#         out = self.relu(out)
#
#         return out
#
#     def forward_withVar(self, x, y):
#         dec, var, entropy, diversity = self.bs.forward_withVar(x, y)
#
#         dec = dec.contiguous()
#
#         residual = x
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         b, c, h, w = x.size()
#         dec1 = dec.view(b, self.div, 1, 1)
#         dec1 = dec1.expand(b, self.div, h, w)
#
#         z = [x, dec1]
#         z1 = [x, 1 - dec1]
#         x = torch.cat(z, 1)
#         x1 = torch.cat(z1, 1)
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         x = self.relu(out)
#
#         out1 = self.conv1(x1)
#         out1 = self.bn1(out1)
#         x1 = self.relu(out1)
#
#         repulsion_loss = torch.max(0.01 - ((x1 - x) ** 2).mean(), Variable(torch.Tensor([0]).type_as(entropy)))
#
#         b, c, h, w = x.size()
#         dec2 = dec.view(b, self.div, 1, 1)
#         dec2 = dec2.expand(b, self.div, h, w)
#
#         z = [x, dec2]
#         z2 = [x, 1 - dec2]
#         x = torch.cat(z, 1)
#         x2 = torch.cat(z2, 1)
#
#         out = self.conv2(x)
#         out = self.bn2(out)
#         x = self.relu(out)
#
#         out2 = self.conv2(x2)
#         out2 = self.bn2(out2)
#         x2 = self.relu(out2)
#         repulsion_loss += torch.max(0.01 - ((x2 - x) ** 2).mean(), Variable(torch.Tensor([0]).type_as(entropy)))
#
#         b, c, h, w = x.size()
#         dec3 = dec.view(b, self.div, 1, 1)
#         dec3 = dec3.expand(b, self.div, h, w)
#
#         z = [x, dec3]
#         z3 = [x, 1 - dec3]
#         x = torch.cat(z, 1)
#         x3 = torch.cat(z3, 1)
#
#         out = self.conv3(x)
#         out = self.bn3(out)
#         out += residual
#         out = self.relu(out)
#
#         out3 = self.conv3(x3)
#         out3 = self.bn3(out3)
#         out3 += residual
#         out3 = self.relu(out3)
#         repulsion_loss += torch.max(0.01 - ((out3 - out) ** 2).mean(), Variable(torch.Tensor([0]).type_as(entropy)))
#
#         return out, var, entropy, diversity, repulsion_loss / 3.0
#
#
# class BasicBlockImageNet_ResDec(nn.Module):
#     expansion = 1
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16, numofclass=10, div=2, binary=False):
#         super(BasicBlockImageNet_ResDec, self).__init__()
#         self.conv1 = conv3x3(inplanes + div, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes + div, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.downsample = downsample
#         self.stride = stride
#
#         self.div = div
#         self.bs = BSLayerWithVar_LargerConv_Conditional(inplanes, planes, stride=stride, reduction=reduction,
#                                                         numofclass=numofclass, div=div, binary=binary)
#
#     def forward(self, x):
#         dec = self.bs(x)
#
#         residual = x
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         b, c, h, w = x.size()
#         dec = dec.contiguous().view(b, self.div, 1, 1)
#         dec1 = dec.expand(-1, -1, h, w)
#         x = torch.cat([x, dec1], 1)
#         out = self.conv1(x)
#         out = self.bn1(out)
#         x = self.relu(out)
#
#         b, c, h, w = x.size()
#         dec2 = dec.expand(-1, -1, h, w)
#         x = torch.cat([x, dec2], 1)
#         out = self.conv2(x)
#         out = self.bn2(out)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
#
#     def forward_withVar(self, x, y):
#         dec, var, entropy, diversity = self.bs.forward_withVar(x, y)
#
#         residual = x
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         b, c, h, w = x.size()
#         dec = dec.contiguous().view(b, self.div, 1, 1)
#         dec1 = dec.expand(b, self.div, h, w)
#         z = [x, dec1]
#         x = torch.cat(z, 1)
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         x = self.relu(out)
#
#         b, c, h, w = x.size()
#         dec2 = dec.expand(b, self.div, h, w)
#         z = [x, dec2]
#         x = torch.cat(z, 1)
#         out = self.conv2(x)
#         out = self.bn2(out)
#
#         out += residual
#         out = self.relu(out)
#
#         return out, var, entropy, diversity
#
#
# class BasicBlockImageNet_ResDec_WithRepul(nn.Module):
#     expansion = 1
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16, numofclass=10, div=2, binary=False):
#         super(BasicBlockImageNet_ResDec_WithRepul, self).__init__()
#         self.conv1 = conv3x3(inplanes + div, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes + div, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.downsample = downsample
#         self.stride = stride
#
#         self.div = div
#         self.bs = BSLayerWithVar_LargerConv_Conditional(inplanes, planes, stride=stride, reduction=reduction,
#                                                         numofclass=numofclass, div=div, binary=binary)
#
#     def forward(self, x):
#         dec = self.bs(x)
#
#         residual = x
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         b, c, h, w = x.size()
#         dec = dec.contiguous().view(b, self.div, 1, 1)
#         dec1 = dec.expand(-1, -1, h, w)
#         x = torch.cat([x, dec1], 1)
#         out = self.conv1(x)
#         out = self.bn1(out)
#         x = self.relu(out)
#
#         b, c, h, w = x.size()
#         dec2 = dec.expand(-1, -1, h, w)
#         x = torch.cat([x, dec2], 1)
#
#         out = self.conv2(x)
#         out = self.bn2(out)
#         out += residual
#         out = self.relu(out)
#
#         return out
#
#     def forward_withVar(self, x, y):
#         dec, var, entropy, diversity = self.bs.forward_withVar(x, y)
#
#         residual = x
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         b, c, h, w = x.size()
#         dec = dec.contiguous().view(b, self.div, 1, 1)
#         dec1 = dec.expand(b, self.div, h, w)
#
#         z = [x, dec1]
#         z1 = [x, 1 - dec1]
#         x = torch.cat(z, 1)
#         x1 = torch.cat(z1, 1)
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         x = self.relu(out)
#
#         out1 = self.conv1(x1)
#         out1 = self.bn1(out1)
#         x1 = self.relu(out1)
#         repulsion_loss = torch.max(0.01 - ((x1 - x) ** 2).mean(), Variable(torch.Tensor([0]).type_as(entropy)))
#
#         b, c, h, w = x.size()
#         dec2 = dec.expand(b, self.div, h, w)
#
#         z = [x, dec2]
#         z2 = [x, 1 - dec2]
#         x = torch.cat(z, 1)
#         x2 = torch.cat(z2, 1)
#
#         out = self.conv2(x)
#         out = self.bn2(out)
#         out += residual
#         out = self.relu(out)
#
#         out2 = self.conv2(x2)
#         out2 = self.bn2(out2)
#         out2 += residual
#         out2 = self.relu(out2)
#         repulsion_loss += torch.max(0.01 - ((out2 - out) ** 2).mean(), Variable(torch.Tensor([0]).type_as(entropy)))
#
#         return out, var, entropy, diversity, repulsion_loss / 2.0
#
#
# def resnet18_imagenet(pretrained=False, **kwargs):
#     model = ResNetImageNet(BasicBlockImageNet, [2, 2, 2, 2], **kwargs)
#
#     return model
#
#
# def resnet18_imagenet_resdec(pretrained=False, **kwargs):
#     model = ResNetImageNet_ResDec(BasicBlockImageNet_ResDec, [2, 2, 2, 2], **kwargs)
#
#     return model
#
#
# def resnet50_imagenet_resdec(pretrained=False, **kwargs):
#     model = ResNetImageNet_ResDec(BottleneckImageNet_ResDec, [3, 4, 6, 3], **kwargs)
#     return model
#
#
# def resnet101_imagenet_resdec(pretrained=False, **kwargs):
#     model = ResNetImageNet_ResDec(BottleneckImageNet_ResDec, [3, 4, 23, 3], **kwargs)
#     return model
#
#
# def resnet34_imagenet(pretrained=False, **kwargs):
#     model = ResNetImageNet(BasicBlockImageNet, [3, 4, 6, 3], **kwargs)
#     return model
#
#
def resnet50_imagenet(pretrained=False, **kwargs):
    model = ResNetImageNet(BottleneckImageNet, [3, 4, 6, 3], **kwargs)
    return model

#
# def resnet101_imagenet(pretrained=False, **kwargs):
#     model = ResNetImageNet(BottleneckImageNet, [3, 4, 23, 3], **kwargs)
#     return model
#
#
# def resnet18_imagenet_resdec_repul(pretrained=False, **kwargs):
#     model = ResNetImageNet_ResDec_Repul(BasicBlockImageNet_ResDec_WithRepul, [2, 2, 2, 2], **kwargs)
#
#     return model
#
#
# def resnet50_imagenet_resdec_repul(pretrained=False, **kwargs):
#     model = ResNetImageNet_ResDec_Repul(BottleneckImageNet_ResDec_WithRepul, [3, 4, 6, 3], **kwargs)
#     return model
