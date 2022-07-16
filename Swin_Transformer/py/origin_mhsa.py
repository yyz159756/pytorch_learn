import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import py.image2emb


class MultiHeadSelfAttention(nn.Module):
    """
    输入input [bs, num_seq, model_dim]
    input 映射 -> q k v [bs, num_seq, model_dim]
    计算 probability
    输出 v * prob, 输出shape:[bs, num_seq, model_dim]
    """
    def __init__(self, model_dim, num_head):
        super(MultiHeadSelfAttention, self).__init__()
        self.model_dim = model_dim
        self.num_head = num_head
        # qkv映射MLP
        self.proj_linear_layer = nn.Linear(model_dim, 3*model_dim)
        self.final_linear_layer = nn.Linear(model_dim, model_dim)

    def forward(self, inp, additive_mask=None):
        bs, seq_len, model_dim = inp.shape
        num_head = self.num_head
        head_dim = model_dim // num_head

        # qkv映射
        proj_output = self.proj_linear_layer(inp)
        # chunk 对最后一维度进行拆分成3份，qkv
        q, k, v = proj_output.chunk(3, dim=-1)  # [bs, T, model_dim]
        # print(q.shape, k.shape, v.shape)

        # 将qkv转为多头形式
        q = q.reshape([bs, seq_len, num_head, head_dim]).transpose(1, 2)
        q = q.reshape([bs*num_head, seq_len, head_dim])

        k = k.reshape([bs, seq_len, num_head, head_dim]).transpose(1, 2)
        k = k.reshape([bs*num_head, seq_len, head_dim])

        v = v.reshape([bs, seq_len, num_head, head_dim]).transpose(1, 2)
        v = v.reshape([bs*num_head, seq_len, head_dim])

        if additive_mask is None:
            prob = torch.bmm(q, k.transpose(-1, -2))
            prob = prob / math.sqrt(head_dim)
            prob = F.softmax(prob, dim=-1)
        else:
            # 对mask扩充num_head倍，因为计算mask没有考虑头数
            additive_mask = additive_mask.tile([num_head, 1, 1])
            prob = torch.bmm(q, k.transpose(-1, -2))
            prob = (prob / math.sqrt(head_dim)) + additive_mask
            prob = F.softmax(prob, dim=-1)

        output = torch.bmm(prob, v)  # [bs*num_head, seq_len, head_dim]
        # 拆开多头
        output = output.reshape([bs, num_head, seq_len, head_dim])
        output = output.transpose(1, 2)
        output = output.reshape([bs, seq_len, model_dim])

        output = self.final_linear_layer(output)

        return prob, output


def test():
    model_dim, num_head = 8, 2
    image_patch_emb, (bs, channel, h, w, patch_size, patch_depth, model_dim_c) = \
        py.image2emb.get_image_test_emb(16, 16)

    model = MultiHeadSelfAttention(model_dim, num_head)
    prob, output = model.forward(image_patch_emb)
    print(output.shape)


if __name__ == '__main__':
    test()

