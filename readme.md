九月份最新CODEs_sp重构代码
[toc]

# 代码结构

代码在code文件夹下.

## 训练分类器作为baseline

train_resnet_baseline.py :用于训练baseline resent18分类器

目前试验结果:

![image-20220907225718471](readme.assets/image-20220907225718471.png)

## 训练ae 重构cifar10数据集

train_ae_with_mseloss.py : 使用mse loss 训练ae重构cifar10数据集

Adam lr 6e-4  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

[199/200] : loss_train= 0.0009195180556229542

## 训练ae生成虚假样本



## 压制训练
