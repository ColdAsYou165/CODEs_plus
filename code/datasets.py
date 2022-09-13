import torch
import torchvision
from utils import transform_train_cifar_miao, transform_test_cifar_miao, transform_only_tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image


def downloadDatasets():
    cifar10_train = torchvision.datasets.CIFAR10(root="../data/cifar10", train=True, download=True,
                                                 transform=transform_train_cifar_miao)
    cifar10_test = torchvision.datasets.CIFAR10(root="../data/cifar10", train=False, download=True,
                                                transform=transform_test_cifar_miao)
    cifar100_train = torchvision.datasets.CIFAR100(root="../data/cifar100", train=True, download=True,
                                                   transform=transform_train_cifar_miao)
    cifar100_test = torchvision.datasets.CIFAR100(root="../data/cifar100", train=False, download=True,
                                                  transform=transform_test_cifar_miao)
    svhn_train = torchvision.datasets.SVHN(root="../data/svhn", split="train", download=True,
                                           transform=transform_train_cifar_miao)
    svhn_test = torchvision.datasets.SVHN(root="../data/svhn", split="test", download=True,
                                          transform=transform_test_cifar_miao)


def chakantupian():
    cifar10_train = torchvision.datasets.CIFAR10(root="../data/cifar10", train=True, download=False,
                                                 transform=transform_train_cifar_miao)
    cifar10_test = torchvision.datasets.CIFAR10(root="../data/cifar10", train=False, download=False,
                                                transform=transform_test_cifar_miao)
    cifar100_train = torchvision.datasets.CIFAR100(root="../data/cifar100", train=True, download=False,
                                                   transform=transform_train_cifar_miao)
    cifar100_test = torchvision.datasets.CIFAR100(root="../data/cifar100", train=False, download=False,
                                                  transform=transform_test_cifar_miao)
    svhn_train = torchvision.datasets.SVHN(root="../data/svhn", split="train", download=False,
                                           transform=transform_train_cifar_miao)
    svhn_test = torchvision.datasets.SVHN(root="../data/svhn", split="test", download=False,
                                          transform=transform_test_cifar_miao)
    loader_cifar10_train = DataLoader(cifar10_train, batch_size=128, shuffle=True, num_workers=2)
    pic, label = next(iter(loader_cifar10_train))
    print(pic.shape)
    save_image(pic, "./pic.jpg")


def downloade_LSUN():
    lsun_train = torchvision.datasets.LSUN(root="../data/lsun", classes="train", transform=transform_train_cifar_miao)
    lsun_test = torchvision.datasets.LSUN(root="../data/lsun", classes="test", transform=transform_test_cifar_miao)
    trainloader_lsun = DataLoader(lsun_train, batch_size=128, shuffle=True, num_workers=4)
    for data, label in trainloader_lsun:
        print(data.shape, label.shape)


if __name__ == "__main__":
    pass
    downloadDatasets()  # 下载数据集
    # chakantupian()
    # downloade_LSUN()
