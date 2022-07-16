import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import py.image2emb
import py.origin_mhsa

import py.window_mhsa
import py.shift_window_mhsa as shift_window_mhsa


class PatchMerging(nn.Module):
    """
    merge_size: 表示将merge_size乘merge_size的patch浓缩乘一个patch，
                假设merge_size=2，则patch数目减少为原来的1/4
    model_dim: 用来构建线性层，线性层参数
    output_depth_scale: 用来构建线性层，线性层参数
        用于构建线性层：
            nn.Linear(
                model_dim * merge_size * merge_size,
                int(model_dim * merge_size * merge_size * output_depth_scale)
            )
    输入： sw_block_out: [bs, num_window, num_patch_in_window, patch_depth]
    输出： output: [bs, T, dim]
        其中：T = num_window * num_patch_in_window / merge_size^2
             dim = patch_depth * merge_size^2 * output_depth_scale

    如何构建Patch Merging?
        1.将window格式的特征转换或图片image格式
            window格式的特征: [bs, num_patch_old, num_patch_in_window_old, patch_depth_old]
            patch格式: [bs, patch_depth_old, height, width]

        2.利用unfold操作，按照merge_size * merge_size的大小得到新的patch,
          形状为[bs, num_patch_new, merge_ size * merge size * patch_depth_old]

        3.使用一个全连接层对depth进行降维成0.5倍，
          也就是从merge_size * merge_size * patch_depth_old映射到
          0.5 * merge_size * merge_size * patch_depth_old

        输出的是patch embedding的形状格式:[bs,num_patch, patch_depth]

        举例说明: 以merge_size=2为例，经过PatchMerging后，
                 patch数目减少为之前的1/4，但是depth增大为原来的2倍
    """
    def __init__(self, model_dim, merge_size, output_depth_scale=0.5):
        super(PatchMerging, self).__init__()
        self.merge_size = merge_size
        self.proj_layer = nn.Linear(
            model_dim * merge_size * merge_size,
            int(model_dim * merge_size * merge_size * output_depth_scale)
        )

    def forward(self, inp):
        """
        input shape: [bs, num_window, num_patch_in_window, patch_depth]
        output shape: [bs, seq_len, dim]
        """
        # window格式转化成image格式
        inp = shift_window_mhsa.window2image(inp)  # [bs, patch_depth, image_h, image_w]

        # 利用卷积的思路将它划分成块
        # merged_window：[bs, num_patch, patch_depth]
        # 如果merge_size=2，则比原来patch缩小4倍，dim扩大4倍
        merged_window = F.unfold(inp, kernel_size=self.merge_size,
                                 stride=self.merge_size).transpose(-1, -2)

        # dim降维 dim=dim*0.5
        # [bs, num_patch, patch_depth*0.5]
        merged_window = self.proj_layer(merged_window)

        return merged_window


def test():
    model_dim, merge_size = 8, 2
    pm = PatchMerging(model_dim, merge_size)
    bs, num_window, num_patch_in_window = 2, 4, 4
    inp = torch.randn(bs, num_window, num_patch_in_window, model_dim)
    merge_output = pm.forward(inp)
    print(inp.shape, "# input.shape")
    print(merge_output.shape, "# output.shape")


if __name__ == '__main__':
    test()
