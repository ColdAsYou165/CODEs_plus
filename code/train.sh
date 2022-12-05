#!/bin/bash
#gpus_list=("0,1" "2,3" "4,5")
#lr_list=(0.00001)
#lr_dis_list=(0.00001 0.0001)
#先只修改w_loss_weight_list吧
#w_loss_weight_list=(0.001 0.0001)
#python train_ae.py --gpus 0,1 --lr 1e-3 --virtual_scale 4 &
#sleep 60
#python train_ae.py --gpus 2,5 --lr 1e-3 --virtual_scale 6 &
#wait
#python train_ae.py --gpus 0,1 --lr 1e-3 --virtual_scale 8 &
#sleep 60
#python train_ae.py --gpus 2,5 --lr 1e-4 --virtual_scale 4 &
##python train_ae.py --gpus 0,1 --lr 1e-5 --mse_loss_weight 2

#python train_ae.py --gpus 0,1 --lr 1e-5 --lr_d 1e-5 --wgan_optim Adam --w_loss_weight 1e-5 --d_loss_real_weight 3 &
#1e-4 1e-3 1e-2 1e-1 1 1e1 1e2 1e3 1e4
#python train_wgan.py --gpus 0 --lr_dis 1e4 --w_loss_weight 1e-4 &
#sleep 60
#python train_wgan.py --gpus 1 --lr_dis 1e4 --w_loss_weight 1e-3 &
#sleep 60
#python train_wgan.py --gpus 2 --lr_dis 1e4 --w_loss_weight 1e-2 &
#sleep 60
#python train_wgan.py --gpus 3 --lr_dis 1e4 --w_loss_weight 1e-1 &
#sleep 60
#python train_wgan.py --gpus 4 --lr_dis 1e4 --w_loss_weight 1 &
#sleep 60
#python train_wgan.py --gpus 5 --lr_dis 1e4 --w_loss_weight 1e1 &
##
#wait
#python train_wgan.py --gpus 0 --lr_dis 1e4 --w_loss_weight 1e2 &
#sleep 60
#python train_wgan.py --gpus 1 --lr_dis 1e4 --w_loss_weight 1e3 &
#sleep 60
#python train_wgan.py --gpus 2 --lr_dis 1e4 --w_loss_weight 1e4 &
#sleep 60
##从这里开始是试一下不用sigmoid
#python train_wgan.py --gpus 3 --lr_dis 1e4 --set_sigmoid False &
#sleep 60
#python train_wgan.py --gpus 4 --le_g 1e-5 --lr_dis 1e4 --set_sigmoid False &
#sleep 60
##微调lr_dis的参数
#python train_wgan.py --gpus 5 --lr_dis 1e4 &
#wait
#python train_wgan.py --gpus 0 --lr_dis 2e4 &
#sleep 60
#python train_wgan.py --gpus 1 --lr_dis 4e4 &
#sleep 60
##换一个更简单的discriminator
#python train_ae_with_3loss_chamfer_w_cross.py --gpus 0 --cross_loss_weight 1e-1 &
#sleep 60
#python train_ae_with_3loss_chamfer_w_cross.py --gpus 1 --cross_loss_weight 1e-2 &
#sleep 60
#python train_ae_with_3loss_chamfer_w_cross.py --gpus 2 --cross_loss_weight 1e-3 &
#sleep 60
#python train_ae_with_3loss_chamfer_w_cross.py --gpus 3 --cross_loss_weight 1e-4 &
#sleep 60
#python train_ae_with_3loss_chamfer_w_cross.py --gpus 4 --cross_loss_weight 1e-5 &
#sleep 60
#python train_ae_with_3loss_chamfer_w_cross.py --gpus 5 --cross_loss_weight 1 &
#sleep 60
#python train_ae.py --gpus 0 --lr 0.1 &
#sleep 60
#python train_ae.py --gpus 1 --lr 0.01 &
#sleep 60
#python train_ae.py --gpus 2 --lr 0.001 &
#sleep 60
#python train_ae.py --gpus 3 --lr 0.0001 &
#sleep 60
#python train_ae.py --gpus 4 --lr 0.00001 &
#sleep 60
#python train_ae.py --gpus 5 --lr 0.000001 &
#sleep 60
#wait
#python train_ae.py --gpus 0 --lr 0.001 --loss_virtual_weight 2 &
#sleep 60
#python train_ae.py --gpus 1 --lr 0.001 --loss_virtual_weight 4 &
#sleep 60
#python train_ae.py --gpus 2 --lr 0.006 --loss_virtual_weight 2 &
#sleep 60
#python train_ae.py --gpus 3 --lr 0.006 --loss_virtual_weight 4 &
#sleep 60
#python train_ae.py --gpus 4 --lr 0.0001 --loss_virtual_weight 2 &
#sleep 60
#python train_ae.py --gpus 5 --lr 0.0001 --loss_virtual_weight 4 &
#sleep 60
#python train_resnet_with_virtual.py --gpus 0 --loss_virtual_weight 0.0009 &
#sleep 60
#python train_resnet_with_virtual.py --gpus 1 --loss_virtual_weight 0.0008 &
#sleep 60
#python train_resnet_with_virtual.py --gpus 2 --loss_virtual_weight 0.0007 &
#sleep 60
#python train_resnet_with_virtual.py --gpus 3 --loss_virtual_weight 0.0006 &
#sleep 60
#python train_resnet_with_virtual.py --gpus 4 --loss_virtual_weight 0.0005 &
#sleep 60
#python train_resnet_with_virtual.py --gpus 5 --loss_virtual_weight 0.0004 &
#sleep 60
#wait
#python train_resnet_with_virtual.py --gpus 0 --loss_virtual_weight 0.0003 &
#sleep 60
#python train_resnet_with_virtual.py --gpus 1 --loss_virtual_weight 0.0002 &
#sleep 60
#python train_resnet_with_virtual.py --gpus 2 --loss_virtual_weight 0.0001 &
#sleep 60
#python train_resnet_with_virtual.py --gpus 3 --loss_virtual_weight 0 &
#sleep 60
#python train_resnet_with_virtual.py --gpus 4 --loss_virtual_weight 0 &
#sleep 60
#python train_resnet_with_virtual.py --gpus 5 --loss_virtual_weight 1e-5 &
#sleep 60
#python train_ae_chamfer_and_crossgan.py --gpus 4 --w_loss_weight 1e-4 &
#sleep 60
#python train_ae_chamfer_and_crossgan.py --gpus 5 --w_loss_weight 1e-5 &
#wait
#python train_ae_chamfer_and_crossgan.py --gpus 4 --w_loss_weight 1e-6 &
#sleep 60
#python train_ae_chamfer_and_crossgan.py --gpus 5 --w_loss_weight 1e-5 --lr_g 2e-5 --lr_d 1e-5 &
#wait
#python train_ae_chamfer_and_crossgan.py --gpus 4 --w_loss_weight 1e-5 --lr_g 2e-5 --lr_d 0 --lr_dis 1e3 &
#sleep 60
#python train_ae_chamfer_and_crossgan.py --gpus 5 --w_loss_weight 1e-5 --lr_g 2e-5 --lr_d 0 --lr_dis 1e4 &
#wait
#python train_ae_chamfer_and_crossgan.py --gpus 4 --w_loss_weight 1e-5 --lr_g 2e-5 --lr_d 0 --lr_dis 1e5 &
#sleep 60
#python train_ae_chamfer_and_crossgan.py --gpus 5 --w_loss_weight 1e-5 --lr_g 0.0006 --lr_d 0 --lr_dis 1e3 &
#wait
#python train_ae_chamfer_and_crossgan.py --gpus 4 --w_loss_weight 1e-5 --lr_g 0.0006 --lr_d 0 --lr_dis 1e3 &
#sleep 60
#python train_ae_chamfer_and_crossgan.py --gpus 5 --w_loss_weight 1e-5 --lr_g 0.0006 --lr_d 0 --lr_dis 1e4 &
#wait
#python train_ae_chamfer_and_crossgan.py --gpus 4 --w_loss_weight 1e-4 --real_cross_d_weight 0.1 &
#sleep 60
#python train_ae_chamfer_and_crossgan.py --gpus 5 --w_loss_weight 1e-5 --real_cross_d_weight 0.001 &
#wait
#python train_resnet_with_virtual.py --gpus 0 --loss_virtual_weight 1 --crossweight 3 &
#sleep 60
#python train_resnet_with_virtual.py --gpus 1 --loss_virtual_weight 1 --crossweight 4 &
#sleep 60
#python train_resnet_with_virtual.py --gpus 2 --loss_virtual_weight 1 --crossweight 5 &
#sleep 60
#python train_resnet_with_virtual.py --gpus 3 --loss_virtual_weight 0.1 --crossweight 3 &
#sleep 60
#python train_resnet_with_virtual.py --gpus 4 --loss_virtual_weight 0.1 --crossweight 4 &
#sleep 60
#python train_resnet_with_virtual.py --gpus 5 --loss_virtual_weight 0.1 --crossweight 5 &
#sleep 60
#wait
#python train_resnet_with_virtual.py --gpus 0 --loss_virtual_weight 5e-3 --crossweight 3 &
#sleep 60
#python train_resnet_with_virtual.py --gpus 1 --loss_virtual_weight 5e-3 --crossweight 4 &
#sleep 60
#python train_resnet_with_virtual.py --gpus 2 --loss_virtual_weight 5e-3 --crossweight 5 &
#sleep 60
#python train_resnet_with_virtual.py --gpus 3 --loss_virtual_weight 5e-4 --crossweight 3 &
#sleep 60
#python train_resnet_with_virtual.py --gpus 4 --loss_virtual_weight 5e-4 --crossweight 4 &
#sleep 60
#python train_resnet_with_virtual.py --gpus 5 --loss_virtual_weight 5e-4 --crossweight 5 &
#sleep 60
#wait
#python train_resnet_with_virtual.py --gpus 0 --loss_virtual_weight 1e-3 --crossweight 3 &
#sleep 60
#python train_resnet_with_virtual.py --gpus 1 --loss_virtual_weight 1e-3 --crossweight 4 &
#sleep 60
#python train_resnet_with_virtual.py --gpus 2 --loss_virtual_weight 1e-3 --crossweight 5 &
#sleep 60
#python train_resnet_with_virtual.py --gpus 3 --loss_virtual_weight 1e-4 --crossweight 3 &
#sleep 60
#python train_resnet_with_virtual.py --gpus 4 --loss_virtual_weight 1e-4 --crossweight 4 &
#sleep 60
#python train_resnet_with_virtual.py --gpus 5 --loss_virtual_weight 1e-4 --crossweight 5 &
#sleep 60

#python train_ae_with_crossloss_and_chamferloss.py --gpus 1 --weight_crossloss 1e-8 &
#sleep 60
#python train_ae_with_crossloss_and_chamferloss.py --gpus 2 --weight_crossloss 1e-7 &
#sleep 60
#python train_ae_with_crossloss_and_chamferloss.py --gpus 3 --weight_crossloss 1e-6 &
#sleep 60
#python train_ae_with_crossloss_and_chamferloss.py --gpus 4 --weight_crossloss 1e-10 &
#sleep 60
#wait
#python train_ae_with_crossloss_and_chamferloss.py --gpus 1 --weight_crossloss 1e-9 &
#sleep 60
#python train_ae_with_crossloss_and_chamferloss.py --gpus 2 --weight_crossloss 1e-14 &
#sleep 60
#python train_ae_with_crossloss_and_chamferloss.py --gpus 3 --weight_crossloss 1e-18 &
#sleep 60
#python train_ae_with_crossloss_and_chamferloss.py --gpus 4 --weight_crossloss 1e-11 &
#sleep 60

#python train_ae_with_3loss_chamfer_w_cross.py --lr_dis 0 --lr_scale 1e4 --cross_loss_weight 1 --gpus 0 &
#sleep 60
#python train_ae_with_3loss_chamfer_w_cross.py --lr_dis 0 --lr_scale 1e4 --cross_loss_weight 1e-1 --gpus 4 &
#sleep 60
#python train_ae_with_3loss_chamfer_w_cross.py --lr_dis 0 --lr_scale 1e4 --cross_loss_weight 1e-2 --gpus 5 &
#sleep 60
#wait
#python train_ae_with_3loss_chamfer_w_cross.py --lr_dis 0 --lr_scale 1e4 --cross_loss_weight 1e-3 --gpus 0 &
#sleep 60
#python train_ae_with_3loss_chamfer_w_cross.py --lr_dis 0 --lr_scale 1e4 --cross_loss_weight 1e-4 --gpus 1 &
#sleep 60
#python train_ae_with_3loss_chamfer_w_cross.py --lr_dis 0 --lr_scale 1e4 --cross_loss_weight 1e-5 --gpus 2 &
#sleep 60
#python train_ae_with_3loss_chamfer_w_cross.py --lr_dis 0 --lr_scale 1e5 --gpus 3 &
#sleep 60
#python train_ae_with_3loss_chamfer_w_cross.py --lr_dis 0 --lr_scale 1e6 --gpus 4 &
#sleep 60
#python train_ae_with_3loss_chamfer_w_cross.py --lr_dis 0 --lr_scale 1e5 --cross_loss_weight 1e-4 --gpus 5 &
#sleep 60
#结果p用没有
#--batch_size  --blend_loss_weight

#python train_ae_with_3loss_chamfer_w_cross_gaijincross.py --gpus 0 --batch_size 128 --blend_loss_weight 1e-1 &
#sleep 60
#python train_ae_with_3loss_chamfer_w_cross_gaijincross.py --gpus 1 --batch_size 128 --blend_loss_weight 1e-2 &
#sleep 60
#python train_ae_with_3loss_chamfer_w_cross_gaijincross.py --gpus 2 --batch_size 128 --blend_loss_weight 1e-3 &
#sleep 60
#python train_ae_with_3loss_chamfer_w_cross_gaijincross.py --gpus 6 --batch_size 128 --blend_loss_weight 1e-4 &
#sleep 60
#python train_ae_with_3loss_chamfer_w_cross_gaijincross.py --gpus 7 --batch_size 128 --blend_loss_weight 1e-5 &
#wait
#
#python train_ae_with_3loss_chamfer_w_cross_gaijincross.py --gpus 0 --batch_size 32 --blend_loss_weight 1e-1 &
#sleep 60
#python train_ae_with_3loss_chamfer_w_cross_gaijincross.py --gpus 1 --batch_size 32 --blend_loss_weight 1e-2 &
#sleep 60
#python train_ae_with_3loss_chamfer_w_cross_gaijincross.py --gpus 2 --batch_size 32 --blend_loss_weight 1e-3 &
#sleep 60
#python train_ae_with_3loss_chamfer_w_cross_gaijincross.py --gpus 6 --batch_size 32 --blend_loss_weight 1e-4 &
#sleep 60
#python train_ae_with_3loss_chamfer_w_cross_gaijincross.py --gpus 7 --batch_size 32 --blend_loss_weight 1e-5 &
#wait
#
#python train_ae_with_3loss_chamfer_w_cross_gaijincross.py --gpus 0 --batch_size 128 --blend_loss_weight 1e-5 &
#sleep 60
#python train_ae_with_3loss_chamfer_w_cross_gaijincross.py --gpus 1 --batch_size 128 --blend_loss_weight 1e-7 &
#sleep 60
#python train_ae_with_3loss_chamfer_w_cross_gaijincross.py --gpus 2 --batch_size 128 --blend_loss_weight 1e-9 &
#sleep 60
#python train_ae_with_3loss_chamfer_w_cross_gaijincross.py --gpus 6 --batch_size 128 --blend_loss_weight 1e-11 &
#sleep 60
#python train_ae_with_3loss_chamfer_w_cross_gaijincross.py --gpus 7 --batch_size 128 --blend_loss_weight 1e-13 &

#python train_ae_with_3loss_chamfer_w_cross_gaijincross.py --gpus 0 --batch_size 128 --blend_loss_weight 1e-1 &
#sleep 60
#python train_ae_with_3loss_chamfer_w_cross_gaijincross.py --gpus 1 --batch_size 128 --blend_loss_weight 1e-2 &
#sleep 60
#python train_ae_with_3loss_chamfer_w_cross_gaijincross.py --gpus 2 --batch_size 128 --blend_loss_weight 1e-3 &
#sleep 60
#python train_ae_with_3loss_chamfer_w_cross_gaijincross.py --gpus 3 --batch_size 128 --blend_loss_weight 1e-4 &
#sleep 60
#python train_ae_with_3loss_chamfer_w_cross_gaijincross.py --gpus 4 --batch_size 128 --blend_loss_weight 1e-5 &
#sleep 60
#python train_ae_with_3loss_chamfer_w_cross_gaijincross.py --gpus 5 --batch_size 128 --blend_loss_weight 1e-6 &
#sleep 60
#python train_ae_with_3loss_chamfer_w_cross_gaijincross.py --gpus 6 --batch_size 128 --blend_loss_weight 1e-7 &
#sleep 60
#python train_ae_with_3loss_chamfer_w_cross_gaijincross.py --gpus 7 --batch_size 128 --blend_loss_weight 1e-8 &
#sleep 60
#wait
#python train_ae_with_3loss_chamfer_w_cross_gaijincross.py --gpus 0 --batch_size 32 --blend_loss_weight 1e-1 &
#sleep 60
#python train_ae_with_3loss_chamfer_w_cross_gaijincross.py --gpus 1 --batch_size 32 --blend_loss_weight 1e-3 &
#sleep 60
#python train_ae_with_3loss_chamfer_w_cross_gaijincross.py --gpus 2 --batch_size 32 --blend_loss_weight 1e-4 &
#sleep 60
#python train_ae_with_3loss_chamfer_w_cross_gaijincross.py --gpus 6 --batch_size 32 --blend_loss_weight 1e-5 &
#sleep 60
#python train_ae_with_3loss_chamfer_w_cross_gaijincross.py --gpus 7 --batch_size 32 --blend_loss_weight 1e-8 &
#sleep 60

#python train_resnet_byvirtual.py --gpus 7 --ae_version 2 --loss_virtual_weight 1e-5 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 3 --ae_version 3 --loss_virtual_weight 1e-5 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 4 --ae_version 4 --loss_virtual_weight 1e-5 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 6 --ae_version 2 --loss_virtual_weight 1 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 0 --ae_version 3 --loss_virtual_weight 1 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 5 --ae_version 4 --loss_virtual_weight 1 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 2 --ae_version 3 --loss_virtual_weight 1e-5 --batch_size 32 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 1 --ae_version 4 --loss_virtual_weight 1e-3 --batch_size 32 &
#sleep 60
#python train_ae_with_3loss_chamfer_w_cross_gaijincross.py --gpus 2 --blend_loss_weight 1 &
#sleep 60
#python train_ae_with_3loss_chamfer_w_cross_gaijincross.py --gpus 3 --blend_loss_weight 1e-3 &
#sleep 60
#python train_ae_with_3loss_chamfer_w_cross_gaijincross.py --gpus 1 --blend_loss_weight 1e-5 &
#sleep 60
#python train_ae_with_3loss_chamfer_w_cross_gaijincross.py --gpus 4 --blend_loss_weight 1e-6 &
#sleep 60
#python train_ae_with_3loss_chamfer_w_cross_gaijincross.py --gpus 5 --blend_loss_weight 1e-5 --lr_scale 1e4 &
#sleep 60
#python train_ae_with_3loss_chamfer_w_cross_gaijincross.py --gpus 7 --blend_loss_weight 1e-5 --lr_scale 1e4 --lr_dis 0 &
#python train_resnet_byvirtual.py --gpus 0 --ae_version 5 --loss_virtual_weight 1 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 1 --ae_version 5 --loss_virtual_weight 1e-1 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 2 --ae_version 5 --loss_virtual_weight 1e-2 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 3 --ae_version 5 --loss_virtual_weight 1e-3 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 4 --ae_version 5 --loss_virtual_weight 1e-4 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 5 --ae_version 5 --loss_virtual_weight 1e-5 &
#wait
#
#
#python train_resnet_byvirtual.py --gpus 0 --ae_version 7 --loss_virtual_weight 1 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 1 --ae_version 7 --loss_virtual_weight 1e-1 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 2 --ae_version 7 --loss_virtual_weight 1e-2 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 3 --ae_version 7 --loss_virtual_weight 1e-3 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 4 --ae_version 7 --loss_virtual_weight 1e-4 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 5 --ae_version 7 --loss_virtual_weight 1e-5 &
#wait
#
#
#python train_resnet_byvirtual.py --gpus 0 --ae_version 6 --loss_virtual_weight 1 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 1 --ae_version 6 --loss_virtual_weight 1e-1 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 2 --ae_version 6 --loss_virtual_weight 1e-2 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 3 --ae_version 6 --loss_virtual_weight 1e-3 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 4 --ae_version 6 --loss_virtual_weight 1e-4 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 5 --ae_version 6 --loss_virtual_weight 1e-5 &
#wait
b=9
#python train_resnet_byvirtual.py --gpus 0 --ae_version $a --loss_virtual_weight 0.005 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 1 --ae_version $a --loss_virtual_weight 0.0045 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 2 --ae_version $a --loss_virtual_weight 0.004 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 3 --ae_version $a --loss_virtual_weight 0.0035 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 4 --ae_version $a --loss_virtual_weight 0.003 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 5 --ae_version $a --loss_virtual_weight 0.0025 &
#sleep 60

##wait
#python train_resnet_byvirtual.py --gpus 5 --ae_version $b --loss_virtual_weight 0.002 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 1 --ae_version $b --loss_virtual_weight 0.0015 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 2 --ae_version $b --loss_virtual_weight 0.001 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 3 --ae_version $b --loss_virtual_weight 0.002 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 4 --ae_version $b --loss_virtual_weight 0.0015 &
#sleep 60
##python train_resnet_byvirtual.py --gpus 5 --ae_version $b --loss_virtual_weight 0.001 &
#a=9
#python train_resnet_byvirtual.py --gpus 0 --ae_version $a --loss_virtual_weight 0.0035 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 1 --ae_version $a --loss_virtual_weight 0.003 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 2 --ae_version $a --loss_virtual_weight 0.0025 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 3 --ae_version $a --loss_virtual_weight 0.002 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 4 --ae_version $a --loss_virtual_weight 0.0015 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 5 --ae_version $a --loss_virtual_weight 0.001 &
#sleep 60
#wait
#python train_resnet_byvirtual.py --gpus 0 --ae_version $a --loss_virtual_weight 0.0035 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 1 --ae_version $a --loss_virtual_weight 0.003 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 2 --ae_version $a --loss_virtual_weight 0.0025 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 3 --ae_version $a --loss_virtual_weight 0.002 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 4 --ae_version $a --loss_virtual_weight 0.0015 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 5 --ae_version $a --loss_virtual_weight 0.001 &
#sleep 60
#wait
#python train_ae_with_3loss_chamfer_w_cross_gaijincross.py --gpus 2 --blend_loss_weight 1e-4 &
#sleep 60
#python train_ae_with_3loss_chamfer_w_cross_gaijincross.py --gpus 3 --blend_loss_weight 1e-6 &
#sleep 60
#python train_ae_with_3loss_generatev2.py --gpus 4 --blend_loss_weight 1e-4 &
#sleep 60
#python train_ae_with_3loss_generatev2.py --gpus 5 --blend_loss_weight 1e-6 &
#python train_ae_with_3loss_generatev3.py --gpus 1 --blend_loss_weight 1e-4 &
#sleep 60
#python train_ae_with_3loss_generatev3.py --gpus 2 --blend_loss_weight 1e-6 &
#sleep 60
#python train_ae_with_3loss_generatev3.py --gpus 3 --lr_g 0.00006 --lr_dis 0.00006 &
#sleep 60
#python train_ae_with_3loss_generatev3.py --gpus 6 --lr_g 0.00006 --lr_dis 0.00006 --blend_loss_weight 1e-4 &
#python train_ae_with_3loss_generatev3.py --gpus 7 --lr_g 0.00006 --lr_dis 0.00006 --blend_loss_weight 1e-6 &
#python train_ae_with_3loss_generatev2.py --gpus 5 &
#python train_ae_with_3loss_generatev2.py --gpus 4 --blend_loss_weight 1e-5 &

#python train_ae_with_3loss_generatev2.py --gpus 0 --w_loss_weight 1e-7 &
#sleep 60
#python train_ae_with_3loss_generatev2.py --gpus 1 --w_loss_weight 1e-9 &
#sleep 60
#python train_resnet_byvirtual_generatev2.py --gpus 2 --ae_version 0 --loss_virtual_weight 1 &
#sleep 60
#python train_resnet_byvirtual_generatev2.py --gpus 3 --ae_version 0 --loss_virtual_weight 0.1 &
#sleep 60
#python train_resnet_byvirtual_generatev2.py --gpus 4 --ae_version 1 --loss_virtual_weight 1 &
#sleep 60
#python train_resnet_byvirtual_generatev2.py --gpus 5 --ae_version 1 --loss_virtual_weight 0.1 &
#wait
#python train_resnet_byvirtual_generatev2.py --gpus 0 --ae_version 2 --loss_virtual_weight 1 &
#sleep 60
#python train_resnet_byvirtual_generatev2.py --gpus 1 --ae_version 2 --loss_virtual_weight 0.1 &
#sleep 60
#python train_resnet_byvirtual_generatev2.py --gpus 2 --ae_version 3 --loss_virtual_weight 1 &
#sleep 60
#python train_resnet_byvirtual_generatev2.py --gpus 3 --ae_version 3 --loss_virtual_weight 0.1 &
#python train_ae_containy.py --gpus 0 --lr 0.06 &
#sleep 60
#python train_ae_containy.py --gpus 1 --lr 0.006 &
#sleep 60
#python train_ae_containy.py --gpus 2 --lr 0.0006 &
#sleep 60
#python train_ae_containy.py --gpus 3 --lr 0.00006 &
#python train_ae_containy.py --gpus 5 --lr 0.00006 --lr_dis 0.00006 &
#sleep 10
#python train_ae_containy.py --gpus 1 --lr 0.0002 --lr_dis 0.0002 &
#sleep 10
#python train_ae_containy.py --gpus 2 --lr 0.00006 --lr_dis 0.0002 &
#sleep 10
#python train_ae_containy.py --gpus 3 --lr 0.00006 --lr_dis 0.00006 --blend_loss_weight 1 &
#sleep 10
#python train_ae_containy.py --gpus 4 --lr 0.00006 --lr_dis 0.00006 --blend_loss_weight 1e-3 &
#wait
#python train_ae_containy.py --gpus 0 --lr 0.00006 --lr_dis 0.00006 --blend_loss_weight 1e-5 &
#sleep 10
#python train_ae_containy.py --gpus 1 --lr 0.00006 --lr_dis 0.00006 --w_loss_weight 1 &
#sleep 10
#python train_ae_containy.py --gpus 2 --lr 0.00006 --lr_dis 0.00006 --w_loss_weight 1e-3 &
#sleep 10
#python train_ae_containy.py --gpus 3 --lr 0.00006 --lr_dis 0.00006 --w_loss_weight 1e-5 &
#sleep 10
#python train_ae_containy.py --gpus 4 --lr 0.0002 --lr_dis 0.0002 --w_loss_weight 1e-5 &
#sleep 10
#python train_ae_containy.py --gpus 5 --lr 0.0002 --lr_dis 0.0002 --w_loss_weight 1e-5 --blend_loss_weight 1e-3 &
#python train_ae_containy.py --gpus 0 --lr 6e-5 --lr_dis 6e-5 &
#sleep 10
#python train_ae_containy.py --gpus 1 --lr 6e-5 --lr_dis 3e-5 &
#sleep 10
#
#python train_ae_containy.py --gpus 3 --lr 2e-2 --lr_dis 2e-2 &
#sleep 10
#python train_ae_containy.py --gpus 4 --lr 2e-4 --lr_dis 2e-4 &
#sleep 10
#python train_ae_containy.py --gpus 5 --lr 2e-4 --lr_dis 1e-4 &
#python train_ae_containy.py --gpus 0 --lr 0.00006 --lr_dis 0.00006 --w_loss_weight 1e-3 &
#sleep 10
#python train_ae_containy.py --gpus 1 --lr 0.00006 --lr_dis 0.00006 --w_loss_weight 1 &
#sleep 10
#python train_ae_containy.py --gpus 2 --lr 0.00006 --lr_dis 0.00006 --w_loss_weight 1e2 &
#sleep 10
#python train_ae_containy.py --gpus 3 --lr 0.00006 --lr_dis 0.00006 --w_loss_weight 1e3 &
#sleep 10
#python train_ae_containy.py --gpus 4 --lr 0.00006 --lr_dis 0.00006 --w_loss_weight 1e3 &
#sleep 10
#python train_ae_containy.py --gpus 5 --lr 0.0002 --lr_dis 0.0002 --w_loss_weight 1 &
#python train_ae_containy.py --gpus 2 --lr 12e-5 --lr_dis 6e-5 &
#sleep 10
#python train_resnet_byvirtual.py --gpus 0 --ae_version 12 --loss_virtual_weight 1 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 1 --ae_version 12 --loss_virtual_weight 1e-1 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 2 --ae_version 12 --loss_virtual_weight 1e-3 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 3 --ae_version 12 --loss_virtual_weight 1e-5 &
##
#python train_resnet_byvirtual.py --gpus 4 --ae_version 13 --loss_virtual_weight 1 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 5 --ae_version 13 --loss_virtual_weight 1e-1 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 7 --ae_version 13 --loss_virtual_weight 1e-3 &
#sleep 60
#wait
#python train_resnet_byvirtual.py --gpus 0 --ae_version 13 --loss_virtual_weight 1e-5 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 4 --ae_version 14 --loss_virtual_weight 1 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 1 --ae_version 14 --loss_virtual_weight 1e-1 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 2 --ae_version 14 --loss_virtual_weight 1e-3 &
#sleep 60
#python train_resnet_byvirtual.py --gpus 3 --ae_version 14 --loss_virtual_weight 1e-5 &
#python train_ae_forkfadernet.py --gpus 0 --method_label 0 --lr 2e-4 --lr_dis 2e-4 &
#sleep 10
#python train_ae_forkfadernet.py --gpus 1 --method_label 0 --lr 2e-4 --lr_dis 2e-4 --cross_loss_weight 1e-3 &
#sleep 10
#python train_ae_forkfadernet.py --gpus 2 --method_label 1 --lr 2e-4 --lr_dis 2e-4 &
#sleep 10
#python train_ae_forkfadernet.py --gpus 3 --method_label 1 --lr 2e-4 --lr_dis 2e-4 --cross_loss_weight 1e-3 &
#sleep 10
#
#python train_ae_forkfadernet.py --gpus 5 --method_label 0 --lr 6e-5 --lr_dis 6e-5 &
#python train_ae_forkfadernet.py --gpus 5 --method_label 0 --lr 6e-5 --lr_dis 6e-5 --cross_loss_weight 1e-3 &
#sleep 10
#python train_ae_forkfadernet.py --gpus 6 --method_label 1 --lr 6e-5 --lr_dis 6e-5 &
#sleep 10
#python train_ae_forkfadernet.py --gpus 7 --method_label 1 --lr 6e-5 --lr_dis 6e-5 --cross_loss_weight 1e-3 &
#python train_ae_containy.py --lr 0.002 --lr_dis 0.002 --gpus 0 &
#sleep 10
#python train_ae_containy.py --lr 0.0002 --lr_dis 0.0002 --gpus 1 &
#sleep 10
#python train_ae_containy.py --lr 0.00002 --lr_dis 0.00002 --gpus 2 &# 这个成了
#sleep 10
#python train_ae_containy.py --lr 0.00002 --lr_dis 0.00002 --gpus 3 &
#python train_resnet_byvirtual.py --gpus 0 --loss_virtual_weight 1 &
#python train_resnet_byvirtual.py --gpus 1 --loss_virtual_weight 0.1 &
#python train_resnet_byvirtual.py --gpus 2 --loss_virtual_weight 0.001 &
#python train_resnet_byvirtual.py --gpus 3 --loss_virtual_weight 1 --batch_size 28 &
#python train_resnet_byvirtual.py --gpus 4 --loss_virtual_weight 1 --batch_size 512 &
#python train_ae_containy.py --gpus 0 --seed 0 &
#python train_ae_containy.py --gpus 1 --seed 1 &
#python train_ae_containy.py --gpus 2 --seed 2 &
#python train_ae_containy.py --gpus 3 --seed 3 &
#python train_ae_containy.py --gpus 4 --seed 42 &
#python train_ae_containy.py --gpus 5 --seed 22 &
#python train_ae_containy.py --gpus 6 --seed 2022 &
#python train_ae_containy.py --gpus 7 --seed 84 &
#python train_resnet_byvirtual.py --gpus 1 --loss_virtual_weight 1 &
#python train_resnet_byvirtual.py --gpus 2 --loss_virtual_weight 0.1 &
#python train_resnet_byvirtual.py --gpus 3 --loss_virtual_weight 1e-3 &
#python train_resnet_byvirtual.py --gpus 0 --loss_virtual_weight 1e-5 &
#python train_ae_containy_withtangloss.py --gpus 1 --nocontent_loss_weight 1e-9 &
#python train_ae_containy_withtangloss.py --gpus 2 --nocontent_loss_weight 1e-13 &
#python train_ae_containy_withtangloss.py --gpus 3 --nocontent_loss_weight 1e-15 &
#python train_ae_containy_withtangloss.py --gpus 4 --nocontent_loss_weight 1e-20 &
#python train_ae_containy_withtangloss.py --gpus 5 --nocontent_loss_weight 1e-7 &
#python train_ae_containy_withtangloss.py --gpus 2 --lr 6e-4 &
#python train_ae_containy_withtangloss.py --gpus 3 --lr 6e-5 &
#python train_ae_containy_withtangloss.py --gpus 3 --lr 6e-3 &
#python train_ae_containy_withtangloss.py --gpus 1 --blend_loss_weight 1 --w_loss_weight 1 &
#python train_ae_containy_withtangloss.py --gpus 1 --blend_loss_weight 1e-3 --w_loss_weight 1 &
#python train_ae_containy_withtangloss.py --gpus 2 --blend_loss_weight 1 --w_loss_weight 1e-3 &
#python train_ae_containy_withtangloss.py --gpus 2 --blend_loss_weight 1 --w_loss_weight 1e-5 &
#python train_ae_containy_withtangloss.py --gpus 3 --blend_loss_weight 1e-5 --w_loss_weight 1 &
#
#python train_ae_containy_withtangloss.py --gpus 3 --blend_loss_weight 1 --w_loss_weight 1 --lr 6e-4 --lr_dis 6e-4 &
#python train_ae_containy_withtangloss.py --gpus 2 --blend_loss_weight 1e-3 --w_loss_weight 1 --lr 6e-4 --lr_dis 6e-4 &
#python train_ae_containy_withtangloss.py --gpus 3 --blend_loss_weight 1 --w_loss_weight 1e-3 --lr 6e-4 --lr_dis 6e-4 &

#
#python train_resnet_by_virtual_generated_by_aecontainywithtangloss.py --ae_version 0 --loss_virtual_weight 1 --gpus 2 &
#python train_resnet_by_virtual_generated_by_aecontainywithtangloss.py --ae_version 1 --loss_virtual_weight 1 --gpus 4 &
#python train_resnet_by_virtual_generated_by_aecontainywithtangloss.py --ae_version 2 --loss_virtual_weight 1 --gpus 5 &
#
#python train_resnet_by_virtual_generated_by_aecontainywithtangloss.py --ae_version 0 --loss_virtual_weight 1e-1 --gpus 2 &
#python train_resnet_by_virtual_generated_by_aecontainywithtangloss.py --ae_version 1 --loss_virtual_weight 1e-1 --gpus 4 &
#python train_resnet_by_virtual_generated_by_aecontainywithtangloss.py --ae_version 2 --loss_virtual_weight 1e-1 --gpus 5 &
#
#python train_resnet_by_virtual_generated_by_aecontainywithtangloss.py --ae_version 0 --loss_virtual_weight 1e-2 --gpus 3 &
#python train_resnet_by_virtual_generated_by_aecontainywithtangloss.py --ae_version 1 --loss_virtual_weight 1e-2 --gpus 4 &
#python train_resnet_by_virtual_generated_by_aecontainywithtangloss.py --ae_version 2 --loss_virtual_weight 1e-2 --gpus 5 &
#
#python train_resnet_by_virtual_generated_by_aecontainywithtangloss.py --ae_version 0 --loss_virtual_weight 1e-3 --gpus 6 &
#wait
#python train_resnet_by_virtual_generated_by_aecontainywithtangloss.py --ae_version 1 --loss_virtual_weight 1e-3 --gpus 2 &
#python train_resnet_by_virtual_generated_by_aecontainywithtangloss.py --ae_version 2 --loss_virtual_weight 1e-3 --gpus 3 &
#
#python train_resnet_by_virtual_generated_by_aecontainywithtangloss.py --ae_version 0 --loss_virtual_weight 1e-5 --gpus 4 &
#python train_resnet_by_virtual_generated_by_aecontainywithtangloss.py --ae_version 1 --loss_virtual_weight 1e-5 --gpus 5 &
#python train_resnet_by_virtual_generated_by_aecontainywithtangloss.py --ae_version 2 --loss_virtual_weight 1e-5 --gpus 6 &
#
#python train_resnet_by_virtual_generated_by_aecontainywithtangloss.py --ae_version 0 --loss_virtual_weight 1 --gpus 2 &
#python train_resnet_by_virtual_generated_by_aecontainywithtangloss.py --ae_version 1 --loss_virtual_weight 1 --gpus 3 &
#
#python train_resnet_by_virtual_generated_by_aecontainywithtangloss.py --ae_version 0 --loss_virtual_weight 1e-1 --gpus 4 &
#python train_resnet_by_virtual_generated_by_aecontainywithtangloss.py --ae_version 1 --loss_virtual_weight 1e-1 --gpus 5 &
#
#python train_resnet_by_virtual_generated_by_aecontainywithtangloss.py --ae_version 0 --loss_virtual_weight 1e-2 --gpus 6 &
#wait
#python train_resnet_by_virtual_generated_by_aecontainywithtangloss.py --ae_version 1 --loss_virtual_weight 1e-2 --gpus 2 &
#
#python train_resnet_by_virtual_generated_by_aecontainywithtangloss.py --ae_version 0 --loss_virtual_weight 1e-3 --gpus 3 &
#python train_resnet_by_virtual_generated_by_aecontainywithtangloss.py --ae_version 1 --loss_virtual_weight 1e-3 --gpus 4 &
#
#python train_resnet_by_virtual_generated_by_aecontainywithtangloss.py --ae_version 2 --loss_virtual_weight 1 --gpus 5 &
#python train_resnet_by_virtual_generated_by_aecontainywithtangloss.py --ae_version 2 --loss_virtual_weight 1e-1 --gpus 6 &
#python train_resnet_by_virtual_generated_by_aecontainywithtangloss.py --ae_version 0 --loss_virtual_weight 1e-5 --gpus 2 &
#python train_resnet_by_virtual_generated_by_aecontainywithtangloss.py --ae_version 2 --loss_virtual_weight 1e-2 --gpus 3 &
#python train_resnet_by_virtual_generated_by_aecontainywithtangloss.py --ae_version 1 --loss_virtual_weight 1e-5 --gpus 4 &
#python train_resnet_by_virtual_generated_by_aecontainywithtangloss.py --ae_version 2 --loss_virtual_weight 1e-3 --gpus 5 &
#python train_resnet_by_virtual_generated_by_aecontainywithtangloss.py --ae_version 2 --loss_virtual_weight 1e-5 --gpus 6 &

#python train_ae_containy_withtangloss.py --gpus 0 --blend_loss_weight 1 --w_loss_weight 1 &
#python train_ae_containy_withtangloss.py --gpus 1 --blend_loss_weight 1e-3 --w_loss_weight 1 &
#python train_ae_containy_withtangloss.py --gpus 2 --blend_loss_weight 1 --w_loss_weight 1e-3 &
#python train_ae_containy_withtangloss.py --gpus 3 --blend_loss_weight 1 --w_loss_weight 1e-5 &
#python train_ae_containy_withtangloss.py --gpus 4 --blend_loss_weight 1e-5 --w_loss_weight 1 &
#python train_ae_containy_withtangloss.py --gpus 5 --blend_loss_weight 1 --w_loss_weight 1 --lr 6e-4 --lr_dis 6e-4 &
#python train_ae_containy_withtangloss.py --gpus 4 --blend_loss_weight 1e-3 --w_loss_weight 1 --lr 6e-4 --lr_dis 6e-4 &
#python train_ae_containy_withtangloss.py --gpus 5 --blend_loss_weight 1 --w_loss_weight 1e-3 --lr 6e-4 --lr_dis 6e-4 &

#python train_resnet_by_virtual_generated_by_aetrainedbychromosome.py --ae_version 0 --scale 2 --gpus 1 &
#python train_resnet_by_virtual_generated_by_aetrainedbychromosome.py --ae_version 0 --scale 8 --gpus 2 &
#
#python train_resnet_by_virtual_generated_by_aecontainywithtangloss.py --ae_version 0 --loss_virtual_weight 1e-1 --gpus 2 &
#python train_resnet_by_virtual_generated_by_aecontainywithtangloss.py --ae_version 1 --loss_virtual_weight 1e-1 --gpus 4 &
#python train_resnet_by_virtual_generated_by_aecontainywithtangloss.py --ae_version 2 --loss_virtual_weight 1e-1 --gpus 5 &
#python train_ae_chromosome.py --gpus 0 --scale 8 &
#python train_ae_chromosome.py --gpus 1 --scale 4 &
#python train_ae_chromosome.py --gpus 2 --scale 2 &
#python train_ae_chromosome.py --gpus 0 --scale 8 --w_loss_weight 1e-1 &
#python train_ae_chromosome.py --gpus 1 --scale 4 --w_loss_weight 1e-1 &
#python train_ae_chromosome.py --gpus 2 --scale 2 --w_loss_weight 1e-1 &
#
#python train_ae_chromosome.py --gpus 4 --scale 4 --w_loss_weight 1e-3 &
#python train_ae_chromosome.py --gpus 5 --scale 2 --w_loss_weight 1e-3 &
#python train_ae_chromosome.py --gpus 4 --scale 4 --w_loss_weight 1e-5 &
#python train_ae_chromosome.py --gpus 5 --scale 2 --w_loss_weight 1e-5 &
#python train_ae_chromosome_blend2tangloss.py --gpus 0 --scale 8 --w_loss_weight 1 &
#python train_ae_chromosome_blend2tangloss.py --gpus 1 --scale 8 --w_loss_weight 1e-3 &
#python train_ae_chromosome_blend2tangloss.py --gpus 2 --scale 8 --w_loss_weight 1e-5 &
#
#python train_ae_chromosome_blend2tangloss.py --gpus 4 --scale 8 --tang_loss_weight 1 &
#python train_ae_chromosome_blend2tangloss.py --gpus 0 --scale 8 --tang_loss_weight 1e-3 &
#python train_ae_chromosome_blend2tangloss.py --gpus 1 --scale 8 --tang_loss_weight 1e-5 &
#
#python train_ae_chromosome_blend2tangloss.py --gpus 2 --scale 16 --w_loss_weight 1e-1 &
#python train_resnet_by_virtual_generated_by_aetrainedbychromosome.py --ae_version 0 --scale 2 --gpus 1 &
#python train_resnet_by_virtual_generated_by_aetrainedbychromosome.py --ae_version 0 --scale 8 --gpus 2 &
#
#python train_resnet_by_virtual_generated_by_aecontainywithtangloss.py --ae_version 3 --loss_virtual_weight 1 --gpus 2 &
#python train_resnet_by_virtual_generated_by_aecontainywithtangloss.py --ae_version 3 --loss_virtual_weight 1e-3 --gpus 4 &
#python train_resnet_by_virtual_generated_by_aecontainywithtangloss.py --ae_version 3 --loss_virtual_weight 1e-5 --gpus 5 &
#
#python train_resnet_by_virtual_generated_by_aetrainedbychromosome.py --ae_version 0 --scale 2 --gpus 0 --loss_virtual_weight 1 &
#python train_resnet_by_virtual_generated_by_aetrainedbychromosome.py --ae_version 1 --scale 2 --gpus 1 --loss_virtual_weight 1 &
#python train_resnet_by_virtual_generated_by_aetrainedbychromosome.py --ae_version 2 --scale 2 --gpus 2 --loss_virtual_weight 1 &
#python train_resnet_by_virtual_generated_by_aetrainedbychromosome.py --ae_version 3 --scale 8 --gpus 3 --loss_virtual_weight 1 &
#wait
#python train_resnet_by_virtual_generated_by_aetrainedbychromosome.py --ae_version 0 --scale 2 --gpus 0 --loss_virtual_weight 1e-3 &
#python train_resnet_by_virtual_generated_by_aetrainedbychromosome.py --ae_version 1 --scale 2 --gpus 1 --loss_virtual_weight 1e-3 &
#python train_resnet_by_virtual_generated_by_aetrainedbychromosome.py --ae_version 2 --scale 2 --gpus 2 --loss_virtual_weight 1e-3 &
#python train_resnet_by_virtual_generated_by_aetrainedbychromosome.py --ae_version 3 --scale 8 --gpus 3 --loss_virtual_weight 1e-3 &
#python train_resnet_by_virtual_generated_by_aetrainedbychromosome.py --ae_version 0 --scale 2 --gpus 3 --loss_virtual_weight 1e-4 &
#python train_resnet_by_virtual_generated_by_aetrainedbychromosome.py --ae_version 1 --scale 2 --gpus 3 --loss_virtual_weight 1e-4 &
#python train_resnet_by_virtual_generated_by_aetrainedbychromosome.py --ae_version 0 --scale 2 --gpus 4 --loss_virtual_weight 1e-5 &
#python train_resnet_by_virtual_generated_by_aetrainedbychromosome.py --ae_version 1 --scale 2 --gpus 4 --loss_virtual_weight 1e-5 &
#python train_resnet_by_normalasvirtual.py --gpus 5 --loss_virtual_weight 1 &
#python train_resnet_by_normalasvirtual.py --gpus 5 --loss_virtual_weight 1e-3 &
#wait
#python train_resnet_by_normalasvirtual.py --gpus 3 --loss_virtual_weight 1e-1 &
#python train_resnet_by_normalasvirtual.py --gpus 4 --loss_virtual_weight 1e-2 &
#python train_resnet_by_normalasvirtual.py --gpus 5 --loss_virtual_weight 1e-4 &
#python train_ae_chromosome_blend2tangloss.py --gpus 0
python train_resnet_by_virtual_generated_by_aecontainywithtangloss.py --gpus 0 --ae_version 0 --loss_virtual_weight 1 &
python train_resnet_by_virtual_generated_by_aecontainywithtangloss.py --gpus 1 --ae_version 1 --loss_virtual_weight 1 &
python train_resnet_by_virtual_generated_by_aecontainywithtangloss.py --gpus 2 --ae_version 0 --loss_virtual_weight 1e-1 &
python train_resnet_by_virtual_generated_by_aecontainywithtangloss.py --gpus 3 --ae_version 1 --loss_virtual_weight 1e-1 &
python train_resnet_by_virtual_generated_by_aecontainywithtangloss.py --gpus 4 --ae_version 0 --loss_virtual_weight 1e-2 &
python train_resnet_by_virtual_generated_by_aecontainywithtangloss.py --gpus 5 --ae_version 1 --loss_virtual_weight 1e-2 &
wait
python train_resnet_by_virtual_generated_by_aecontainywithtangloss.py --gpus 0 --ae_version 0 --loss_virtual_weight 1e-4 &
python train_resnet_by_virtual_generated_by_aecontainywithtangloss.py --gpus 1 --ae_version 1 --loss_virtual_weight 1e-4 &
python train_resnet_by_virtual_generated_by_aecontainywithtangloss.py --gpus 2 --ae_version 0 --loss_virtual_weight 1e-5 &
python train_resnet_by_virtual_generated_by_aecontainywithtangloss.py --gpus 3 --ae_version 1 --loss_virtual_weight 1e-5 &
