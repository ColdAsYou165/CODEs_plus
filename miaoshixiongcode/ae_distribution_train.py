
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision import datasets
from AutoEncoder.autoencoder import AutoEncoder
from Autoencoder_HC.models import SegNet
import slice_recom
import argparse
import chamfer3D.dist_chamfer_3D
# from Autoencoder_HC.models import SegNet


parser = argparse.ArgumentParser()
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0001')
parser.add_argument('--lr_d', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--dataset_type', type=str, default='imagenet150k')
parser.add_argument('--data_dir', type=str, default='/home/user/MySSDWork/MiaoData/Mdata/datasets/ImageNet-150K')
parser.add_argument('--ae_type', type=str, default='SN')
parser.add_argument('--epochs', type=int, default=3001)
parser.add_argument('--dweight', type=float, default=0.00001)
parser.add_argument('--block_size', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=32)

opt = parser.parse_args()


if opt.ae_type == 'AE':
    autoencoder = AutoEncoder()
elif opt.ae_type == 'SN':
    autoencoder = SegNet()
autoencoder.cuda()
autoencoder=nn.DataParallel(autoencoder)
autoencoder.train()
optimizer_ae = torch.optim.Adam(autoencoder.parameters(), lr=opt.lr)

# Data
print('==> Preparing data..')
# cifar
transform_train_cifar = transforms.Compose([

    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test_cifar = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_base = [transforms.ToTensor()]
transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=2),
    ] + transform_base)
transform_test_M = transforms.Compose(transform_base)
    

    
transform_train_M = transforms.RandomChoice([transform_train, transform_test_M])
if opt.dataset_type == 'cifar100':
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train_cifar)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test_cifar)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
elif opt.dataset_type == 'cifar10':
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train_cifar)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test_cifar)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
elif opt.dataset_type == 'mnist':
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train_M)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test_M)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
elif opt.dataset_type == 'fmnist':
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train_M)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test_M)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)    
elif opt.dataset_type == 'svhn':
    transform_base = [transforms.ToTensor()]
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='edge'),
    ] + transform_base)
    transform_test = transforms.Compose(transform_base)
    
    transform_train = transforms.RandomChoice([transform_train, transform_test])
    trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_train,target_transform=None)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test,target_transform=None)
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
# elif opt.dataset_type == 'svhn':
#     trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_train_cifar,
#                                          target_transform=None)
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

#     testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test_cifar,
#                                         target_transform=None)
#     testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
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


    DataFolderT = '/ssd/KKtangData/ImageNet/Train'  # '/home/Tang/DataSet/ImageNet/train'  #'/home/KekeOthers/DataSet/ImageNet-150K/train'
    DataFolderV = '/ssd/KKtangData/ImageNet/Val'  # '/home/Tang/DataSet/ImageNet/val'   #'/home/KekeOthers/DataSet/ImageNet-150K/val'

    trainset = torchvision.datasets.ImageFolder(DataFolderT, transform=imagenet_train_transform)
    testset = torchvision.datasets.ImageFolder(DataFolderV, transform=imagenet_val_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=12, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False, num_workers=10)
elif opt.dataset_type == 'tiny_imagenet':
    normalize_imagenet = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
    tiny_imagenet_train_transform = transforms.Compose([
        # transforms.Resize(size=(32, 32)),
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # normalize_imagenet
    ])

    imagenet_val_transform = transforms.Compose([
        # transforms.Resize(size=(32, 32)),
        transforms.ToTensor(),
        # normalize_imagenet
    ])
    trainset = torchvision.datasets.ImageFolder(opt.oc101_dir + '/train', transform=tiny_imagenet_train_transform)
    testset = torchvision.datasets.ImageFolder(opt.oc101_dir + '/val', transform=imagenet_val_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

elif opt.dataset_type=='101OC':
    data_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])
    oc_dataset = datasets.ImageFolder(root=opt.oc101_dir,
                                      transform=data_transform)
    trainloader = torch.utils.data.DataLoader(oc_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=10)
elif opt.dataset_type=='imagenet150k':


    data_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])
    oc_dataset = datasets.ImageFolder(root=opt.data_dir+'/train',
                                      transform=data_transform)
    trainloader = torch.utils.data.DataLoader(oc_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)

ndf = 64

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

ngpu = int(opt.ngpu)
nc = 3
nz = 100
ngf = 64
if opt.dataset_type == '101OC' or opt.dataset_type == 'imagenet150k':

    class Discriminator(nn.Module):
        def __init__(self, ngpu):
            super(Discriminator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is (nc) x 64 x 64  32   128  
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32  16    64
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16 8      32
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8 4    16
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # 8
                nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 16),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4    4
                # nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False),
                # nn.BatchNorm2d(ndf * 32),
                # nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False)


        )

        def forward(self, input):
            if input.is_cuda and self.ngpu > 1:
                output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            else:
                output = self.main(input)
            # print('-----------------------------')
            # print(output.shape)
            output = output.mean(0)
            # return output.view(-1, 1).squeeze(1)
            return output.view(1)
elif opt.dataset_type=='tiny_imagenet':
    class Discriminator(nn.Module):
        def __init__(self, ngpu):
            super(Discriminator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is (nc) x 64 x 64  32   128
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32  16    64
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16 8      32
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8 4    16
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # 8
                # state size. (ndf*8) x 4 x 4    4
                # nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False),
                # nn.BatchNorm2d(ndf * 32),
                # nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)

            )

        def forward(self, input):
            if input.is_cuda and self.ngpu > 1:
                output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            else:
                output = self.main(input)
            # print('-----------------------------')
            # print(output.shape)
            output = output.mean(0)
            # return output.view(-1, 1).squeeze(1)
            return output.view(1)

elif opt.dataset_type=='cifar10' or opt.dataset_type == 'cifar100' or opt.dataset_type == 'svhn':
    class Discriminator(nn.Module):
        def __init__(self, ngpu):
            super(Discriminator, self).__init__()
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

        def forward(self, input):
            if input.is_cuda and self.ngpu > 1:
                output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            else:
                output = self.main(input)
            # print('-----------------------------')
            # print(output.shape)
            output = output.mean(0)
            # return output.view(-1, 1).squeeze(1)
            return output.view(1)
elif opt.dataset_type == 'mnist':
    ndf=512
    nc=3
    class Discriminator(nn.Module):
        def __init__(self, ngpu):
            super(Discriminator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is (nc) x 64 x 64  32
                nn.Conv2d(nc, ndf, 3, 2, 1, bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32  16
                nn.Conv2d(ndf, 256, 3, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16 8
                nn.Conv2d(256, 128, 3, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.AvgPool2d(4),
                # state size. (ndf*4) x 8 x 8 4
                # nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                # nn.BatchNorm2d(ndf * 8),
                # # nn.LeakyReLU(0.2, inplace=True),
                # # state size. (ndf*8) x 4 x 4
                # nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),

            )
            self.fc = nn.Sequential(
            # reshape input, 128 -> 1
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        def forward(self, x, y=None):
            y_ = self.main(x)
            y_ = y_.view(y_.size(0), -1)
            y_ = self.fc(y_)
            return y_


netD = Discriminator(ngpu).cuda()
netD.apply(weights_init)

real_label = 1
fake_label = 0
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))

chamLoss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()

def ae(epoch):
    one = torch.FloatTensor([1])
    mone = one * -1
    one, mone = one.cuda(), mone.cuda()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # inputs=inputs.expand(-1,3,-1,-1)
        # print(inputs.size())
        netD.zero_grad()
        real_cpu = inputs.cuda()
        batch_size = real_cpu.size()[0]
        output = netD(real_cpu)
        d_loss_real = output
        d_loss_real.backward(one)
        D_x = output.mean().item()
        
        inputs_adv = slice_recom.get_com_img(inputs, opt.block_size, opt.block_size)
        inputs_adv = inputs_adv.cuda()
        
        decoded = autoencoder(inputs_adv) 
        # with torch.no_grad():
        #     decoded_1=decoded.clone()
        output = netD(decoded.detach())
        d_loss_fake = output
        d_loss_fake.backward(mone)
        D_G_z1 = output.mean().item()
        # errD = errD_real + errD_fake
        errD = d_loss_fake - d_loss_real
        # Update D
        optimizerD.step()
       
        autoencoder.zero_grad()
       
        output = netD(decoded)
        g_loss = opt.dweight * output
        g_loss.backward(one,retain_graph=True)
       
        inputs_adv_re = inputs_adv.transpose(1, 3).transpose(1, 2)
        if opt.dataset_type == '101OC' or opt.dataset_type == 'imagenet150k':
            inputs_adv_re = torch.reshape(inputs_adv_re, [batch_size, 16384, 3])
            decoded_re = decoded.transpose(1, 3).transpose(1, 2)  
            decoded_re = torch.reshape(decoded_re, [batch_size, 16384, 3])
        elif opt.dataset_type=='tiny_imagenet':
            inputs_adv_re = torch.reshape(inputs_adv_re, [batch_size, 4096, 3])
            decoded_re = decoded.transpose(1, 3).transpose(1, 2)  
            decoded_re = torch.reshape(decoded_re, [batch_size, 4096, 3])
        elif opt.dataset_type=='cifar10' or opt.dataset_type=='cifar100' or opt.dataset_type=='svhn':
            inputs_adv_re = torch.reshape(inputs_adv_re, [batch_size, 1024, 3])
            decoded_re = decoded.transpose(1, 3).transpose(1, 2)  
            decoded_re = torch.reshape(decoded_re, [batch_size, 1024, 3])
        else:
            inputs_adv_re = torch.reshape(inputs_adv_re, [batch_size, 28*28, 3])
            decoded_re = decoded.transpose(1, 3).transpose(1, 2)  
            decoded_re = torch.reshape(decoded_re, [batch_size, 28*28, 3])
            
        dist1, dist2, _, _ = chamLoss(inputs_adv_re, decoded_re)
        errG = (torch.mean(dist1)) + (torch.mean(dist2))
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizer_ae.step()
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f errG: %.4f'
              % (epoch, opt.epochs - 1, batch_idx, len(trainloader),
                 errD.item(), g_loss.item(), D_x, D_G_z1, D_G_z2, errG))
    if epoch % 200 == 0:
        torch.save(autoencoder.state_dict(),
                   './checkpoint/ae-' + opt.dataset_type + '-epoch' + str(epoch) + 'dweight' + str(
                       opt.dweight) + 'bs' + str(opt.block_size) + '_new.pth')
        vutils.save_image(decoded, './test_img/ae/ae-' + opt.dataset_type + '-epoch' + str(epoch) + '-dweight' + str(
            opt.dweight) + '-bs' + str(opt.block_size)+'-lr'+str(opt.lr) + '-lr_d'+str(opt.lr_d)+'re.jpg')
        vutils.save_image(inputs_adv, './test_img/slice/ae-' + opt.dataset_type + '-epoch' + str(epoch) + '-dweight' + str(
            opt.dweight) + '-bs' + str(opt.block_size)+'-lr'+str(opt.lr) + '-lr_d'+str(opt.lr_d)+ 'slice.jpg')
        vutils.save_image(real_cpu, './test_img/orig/ae-' + opt.dataset_type + '-epoch' + str(epoch) + '-dweight' + str(
            opt.dweight) + '-bs' + str(opt.block_size)+'-lr'+str(opt.lr) + '-lr_d'+str(opt.lr_d)+'org.jpg')

if __name__ == '__main__':
    # train
    for epoch in range(1, opt.epochs):
        ae(epoch)

