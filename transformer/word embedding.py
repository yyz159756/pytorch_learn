# %%
from pyexpat import model
from turtle import pos
import numpy
import torch as T
import torch.nn as nn
import torch.nn.functional as F
# %%
# 假设有两个句子
batch_size = 2
# 每个句子长度为2~5
src_len = T.randint(2, 5, (batch_size, ))
tgt_len = T.randint(2, 5, (batch_size, ))
print(src_len)
print(tgt_len)
# 方便研究，我们写死
src_len = T.Tensor([2, 4]).to(T.int32)
tgt_len = T.Tensor([4, 3]).to(T.int32)
print(src_len)
print(tgt_len)
# %%
# 单词表大小
max_source_word_num = 8
max_target_word_num = 8
# 最大序列长度
max_source_seq_len = 5
max_target_seq_len = 5
# 生成seq
src_seq = [T.randint(1, max_source_word_num, (L,)) for L in src_len]
# padding
src_seq = list(map(lambda x: F.pad(x, (0, max_source_seq_len - len(x))), src_seq))
# 升一维方便我们拼接
src_seq = list(map(lambda x: T.unsqueeze(x, 0), src_seq))
# 拼接
src_seq = T.cat(src_seq, 0)
print(src_seq)

tgt_seq = [F.pad(T.randint(1, max_target_word_num, (L,)), (0, max_target_seq_len-L)) for L in tgt_len]
tgt_seq = list(map(lambda x: T.unsqueeze(x, 0), tgt_seq))
tgt_seq = T.cat(tgt_seq, 0)
print(tgt_seq)


# %%

# %%
model_dim = 8
src_embedding_table = nn.Embedding(max_source_word_num + 1, model_dim)
tgt_embedding_table = nn.Embedding(max_target_word_num + 1, model_dim)
print(src_embedding_table.weight.size())

# 测试一下forward
src_embedding = src_embedding_table(src_seq)
print(src_embedding.size())
# %%

# %%
max_position_len = 5
pos_matrix = T.arange(max_position_len).reshape((-1, 1))
print(pos_matrix)

# 因为要分奇数列和偶数列，所以间隔为2
i_matrix = T.pow(10000, T.arange(0, model_dim, 2).reshape([1, -1]) / model_dim)
print(i_matrix)
# 构建embedding矩阵
pe_embedding_table = T.zeros([max_position_len, model_dim])
# 偶数列，行不变，0：：2偶数列，意思是下标从0开始，直到最后，取步长为2的所有元素
pe_embedding_table[:, 0::2] = T.sin(pos_matrix / i_matrix)
# 奇数列
pe_embedding_table[:, 1::2] = T.cos(pos_matrix / i_matrix)
print(pe_embedding_table)
# %%
# 改写nn Module weight方式创建pe embedding
pe_embedding = nn.Embedding(max_position_len, model_dim)
pe_embedding.weight = nn.Parameter(pe_embedding_table, requires_grad=False)
print(pe_embedding.weight.size())


# %%
# 构造位置索引
src_pos = T.cat([T.unsqueeze(T.arange(max_position_len), 0) for _ in src_len] , 0)
print(src_pos)

tgt_pos = T.cat([T.unsqueeze(T.arange(max_position_len), 0) for _ in tgt_len] , 0)
# forward 前向计算src-pe
src_pe_embedding = pe_embedding(src_pos)
print(src_pe_embedding.size())
