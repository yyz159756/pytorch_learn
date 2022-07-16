import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import py.origin_mhsa
import py.image2emb


def window_multi_head_self_attention(
        patch_emb,
        mhsa,
        window_size,
):
    """
    patch_emb: 输入： [bs, num_patch, patch_depth]
    mhsa: 实例化好的多头自注意力机制
    window_size: window边长
    输出：[bs, num_window, num_patch_in_window, patch_dim]
        其中 num_window, num_patch_in_window = num_patch
    """
    num_patch_in_window = window_size * window_size
    bs, num_patch, patch_depth = patch_emb.shape
    image_h = image_w = int(math.sqrt(num_patch))

    # 将patch emb转化为image形式
    patch_emb = patch_emb.transpose(-1, -2)
    # 图片形式，patch_depth视为channel
    patch = patch_emb.reshape([bs, patch_depth, image_h, image_w])
    # window: [bs, num_window = (image_h/4)^2, window_depth = window_size^2 * patch_depth]
    window = F.unfold(patch, kernel_size=window_size, stride=window_size)\
        .transpose(-1, -2)

    bs, num_window, patch_depth_times_num_patch_in_window = window.shape
    # 因为窗和窗之间是独立的，bs可以和窗数量维度合并
    # window [bs*num_window, num_patch_in_window, patch_depth]
    window = window.reshape([bs*num_window, patch_depth,
                             num_patch_in_window]).transpose(-1, -2)

    # 计算window内的多头自注意力机制
    attention_prob, output = mhsa(window)
    # 拆成4维的window格式
    output = output.reshape([bs, num_window, num_patch_in_window, patch_depth])

    return attention_prob, output


def test():
    model_dim, num_head = 8, 2
    image_patch_emb, (bs, channel, h, w, patch_size, patch_depth, model_dim_c) = \
        py.image2emb.get_image_test_emb(16, 16)  # emb: [bs, num_patch, model_dim=8]
    print(image_patch_emb.shape, "# image_patch_emb.shape")

    window_size = 2

    mhsa = py.origin_mhsa.MultiHeadSelfAttention(model_dim, num_head)
    prob, w_mhsa_output = window_multi_head_self_attention(image_patch_emb, mhsa, window_size)
    print(w_mhsa_output.shape, "# w_mhsa_output.shape")


if __name__ == '__main__':
    test()

