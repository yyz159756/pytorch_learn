import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def image2emb_conv(image, kernel):
    """
    如何基于图片生成patch embedding?

    方法二
    patch_depth = channel × patch_size × patch_size

    · model_dim_C看作输出通道数，设置kernel为[model_dim_C, input_channel, patch_size, patch_size]

    · 调用PyTorch的conv2d API得到卷积的输出张量，形状为[bs, output_channel, patch_size, patch_size]

    · 转换为[bs, num_patch, model_dim_C]的格式，即为patch embedding
    """
    # image shape [bs, ic, h, w]
    # kernel shape [model_dim_C, input_channel, patch_size, patch_size]
    stride = kernel.shape[0]
    output = F.conv2d(image, kernel, stride=stride)  # [bs, oc, oh, ow]
    bs, oc, oh, ow = output.shape
    patch_emb = output.reshape([bs, oc, oh*ow]).transpose(-1, -2)
    return patch_emb


def image2emb_naive(image, patch_size, weight):
    """
    如何基于图片生成patch embedding?

    方法一
    · 基于pytorch unfold的API来将图片进行分块，也就是模仿卷积矩阵乘法的思路，
      设置kernel_size=stride=patch_size，得到分块后的图片就是没有交叠的，
      得到格式为[bs, num_patch, patch_depth]的张量
      patch_depth = channel × patch_size × patch_size

    · 得到张量以后将张量与形状为[patch_depth, model_dim_C)的权重矩阵进行乘法操作，
      即可得到形状为[bs, num_patch, model_dim_C)的patch embedding
    """
    # image shape [bs, c, h, w]
    # F.unfold 出来的结果shape [bs, patch_depth, num_patch]
    # 转置一下得到patch shape [bs, num_patch, patch_depth]
    # patch_depth = patch_size * patch_size * in_channel
    patch = F.unfold(image, kernel_size=(patch_size, patch_size),
                     stride=(patch_size, patch_size)).transpose(-1, -2)

    patch_emb = patch @ weight  # [bs, num_patch, model_dim]
    return patch_emb


def get_image_test_emb(image_h, image_w, bs=2, channel=3, patch_size=2,
                       model_dim=8):
    bs, channel, h, w = bs, channel, image_h, image_w
    patch_size = patch_size  # ?×?的像素点当作一块区域patch
    patch_depth = patch_size * patch_size * channel  # 48
    model_dim_c = model_dim
    weight = nn.Parameter(torch.randn(patch_depth, model_dim_c))
    image = torch.randn(bs, channel, h, w)
    emb = image2emb_naive(image, patch_size, weight)
    return emb, (bs, channel, h, w, patch_size, patch_depth, model_dim_c)


def test():
    emb, _ = get_image_test_emb(16, 16)
    print(emb.shape)


if __name__ == '__main__':
    test()
