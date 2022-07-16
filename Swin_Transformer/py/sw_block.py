import torch.nn as nn
import py.image2emb
import py.origin_mhsa
import py.window_mhsa
from py.shift_window_mhsa import shift_window_multi_head_self_attention


class SwinTransformerBlock(nn.Module):
    """
    每个MLP包含两层，分别是4 * model_dim和model_dim的大小，
    先映射到大的维度上，再还原到原来的维度

    输入：patch embedding格式
        [bs, num_patch, patch_depth]

    输出：window的数据格式
        [bs, num_window, num_patch_in_window, patch_depth]
        其中：
            num_window * num_patch_in_window = num_patch
    """
    def __init__(self, model_dim, window_size, num_head):
        super(SwinTransformerBlock, self).__init__()
        self.window_size = window_size
        self.layer_norm1 = nn.LayerNorm(model_dim)
        self.layer_norm2 = nn.LayerNorm(model_dim)
        self.layer_norm3 = nn.LayerNorm(model_dim)
        self.layer_norm4 = nn.LayerNorm(model_dim)

        self.wsma_mlp1 = nn.Linear(model_dim, 4*model_dim)
        self.wsma_mlp2 = nn.Linear(4*model_dim, model_dim)
        self.swsma_mlp1 = nn.Linear(model_dim, 4*model_dim)
        self.swsma_mlp2 = nn.Linear(4*model_dim, model_dim)
        # 一个window的mhsa，一个shifted window的mhsa
        self.mhsa1 = py.origin_mhsa.MultiHeadSelfAttention(model_dim, num_head)
        self.mhsa2 = py.origin_mhsa.MultiHeadSelfAttention(model_dim, num_head)

    def forward(self, inp):
        bs, num_patch, patch_depth = inp.shape

        '''block第一层'''
        # 首先层归一化
        inp1 = self.layer_norm1(inp)
        # 送进mhsa
        prob, w_msa_output = py.window_mhsa.window_multi_head_self_attention\
            (inp1, self.mhsa1, window_size=self.window_size)
        # 获取num_window和num_patch_in_window
        bs, num_window, num_patch_in_window, patch_depth =\
            w_msa_output.shape
        # 做一个残差连接
        w_msa_output = inp1 + w_msa_output.reshape([bs, num_patch, patch_depth])
        # w层归一化和两层MLP
        output1 = self.wsma_mlp2(self.wsma_mlp1(self.layer_norm2(w_msa_output)))
        # 再经过残差连接
        output1 += w_msa_output

        '''block第二层'''
        # 经过层归一化
        input2 = self.layer_norm3(output1)
        input2 = input2.reshape([bs, num_window, num_patch_in_window, patch_depth])
        # 送入shift window mhsa
        sw_msa_output = shift_window_multi_head_self_attention(input2,
                                                               self.mhsa2,
                                                               window_size=self.window_size,
                                                               )
        # 经过残差连接
        sw_msa_output = output1 + sw_msa_output.reshape([bs,
                                                         num_patch,
                                                         patch_depth])
        # 层归一化和两层MLP
        output2 = self.swsma_mlp2(self.swsma_mlp1(self.layer_norm4(sw_msa_output)))
        # 残差连接
        output2 += sw_msa_output

        output2 = output2.reshape([bs, num_window, num_patch_in_window,
                                   patch_depth])

        return output2


def test():
    model_dim, num_head = 8, 2
    window_size = 4
    image_patch_emb, (bs, channel, h, w, patch_size, patch_depth, model_dim_c) = \
        py.image2emb.get_image_test_emb(256, 256)  # emb: [bs, num_patch, model_dim=8]
    print(image_patch_emb.shape, "# image_patch_emb.shape")

    block = SwinTransformerBlock(model_dim, window_size, num_head)
    output = block(image_patch_emb)

    print(output.shape, "# sw_block.shape")


if __name__ == '__main__':
    test()