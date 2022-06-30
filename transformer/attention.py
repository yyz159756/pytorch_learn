# %%
import torch
import torch.nn.functional as F
import math


# %% 注意机制实现
def attention(query: torch.Tensor, key: torch.Tensor,
              value: torch.Tensor, mask=None, dropout=None):
    """
    shape: [batch_size * num_head, seq_len, model_dim/num_head]
    """
    # size()的最后一个维度
    d_k = query.size(-1)
    # 计算attention
    scores = torch.matmul(query, key.transpose(-2, -1)) \
        / math.sqrt(d_k)
    if mask is not None:
        # mask矩阵等于0的地方，将该元素置为负无穷
        scores = scores.masked_fill(mask == 0, -1e9)
    # 计算softmax，因为softmax是单调函数，
    # 又因为mask的地方被置为了负无穷，所以softmax出来是0
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


if __name__ == '__main__':
    # 测试
    x = torch.ones([2, 2, 3])
    tmp = attention(x, x, x)
    print(tmp)
