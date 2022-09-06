'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


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


class VGG(nn.Module):
    def __init__(self, vgg_name,class_nums):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, class_nums)

    def _make_layers(self, cfg):
        # print('_make_layers',adv)
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                # print('m',adv)
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                # print('nom',adv)
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           Mixturenorm2d(x, adv=False),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.ModuleList(layers)

    def forward(self, out, adv=False):
        # print(adv)
        for layer in self.features:
           if isinstance(layer,Mixturenorm2d):
               out=layer(out,adv)
           else:
               out=layer(out)

        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


# net = VGG('VGG11')
# x = torch.randn(2, 3, 32, 32)
# y = net(x, adv=True)
# print(y.size())

# test()
