
import numpy as np
import torch

def get_com_img(x, h, w):
    # x b c h w
    # np b h w c
    x = x.numpy()
    x = np.transpose(x, [0, 2, 3, 1])
    shape_list = list(x.shape)
    block_num = [h, w]
    block_len = [int(shape_list[1] / block_num[0]), int(shape_list[2] / block_num[1])]
    channels = shape_list[-1]
    BB = np.reshape(x, [-1, block_num[0], block_len[0], block_num[1], block_len[1], channels])
    BB = np.transpose(BB, [0, 1, 3, 2, 4, 5])
    BB = np.reshape(BB, [-1, block_len[0], block_len[1], channels])
    np.random.shuffle(BB)
    BB = np.reshape(BB, [-1] + block_num + block_len + [channels])
    BB = np.transpose(BB, [0, 1, 3, 2, 4, 5])
    BB = np.reshape(BB, [-1] + shape_list[1:])
    BB = np.transpose(BB, [0, 3, 1, 2])
    BB = torch.tensor(BB, requires_grad=True, dtype=torch.float32)
    return BB

# def get_com_img(x, h, w):
#     # shape_list = x.get_shape().as_list()
#     # x b c h w
#     # np b h w c
#     x = x.numpy()
#     x = np.transpose(x, [0, 2, 3, 1])
#     shape_list = list(x.shape)
#     block_num = [h, w]
#     block_len = [int(shape_list[1] / block_num[0]), int(shape_list[2] / block_num[1])]
#     channels = shape_list[-1]
#     BB = np.reshape(x, [-1, block_num[0], block_len[0], block_num[1], block_len[1], channels])
#     BB = np.transpose(BB, [1, 3, 2, 4, 5, 0])
#     BB = np.reshape(BB, [ shape_list[0]*block_num[0] * block_num[1], block_len[0], block_len[1], channels])
#     np.random.shuffle(BB)
#     BB = np.reshape(BB,  [shape_list[0]]+block_num + block_len + [channels])
#     BB = np.transpose(BB, [ 0, 2, 1, 3, 4,5])
#     BB = np.reshape(BB, [shape_list[0]] + shape_list[1:])
#     BB = np.transpose(BB, [0, 3, 1, 2])
#     BB = torch.tensor(BB, requires_grad=True, dtype=torch.float32)
#     return BB

