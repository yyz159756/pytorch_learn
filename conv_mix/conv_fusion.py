# %%
import sys
import torch

import torch.nn.functional as F
import torch.nn as nn

# %%
in_channels = 2
ou_channels = 2
kernel_size = 3
w = 9 
h = 9

# %%
conv_2d = nn.Conv2d(in_channels, ou_channels, kernel_size, padding='same')
conv_2d.weight.size()

# %%
conv_2d_pointwise = nn.Conv2d(in_channels, ou_channels, kernel_size=1)
conv_2d_pointwise.weight.size()

# %%
x = torch.ones(1, in_channels, w, h) # batchsize=1，channels, wide, height
x.size()

# %%
result1 = conv_2d(x) + conv_2d_pointwise(x) + x
result1.size()


# %%
# 改成3*3卷积形式
# point-wise和x写成3*3卷积
# 把3个卷积融合成一个卷积
# F.pad 操作把 [3,3,1,1]变成[3,3,3,3] 
# 从里到外每个dim两个参数，一个上下一个左右，每个元素的上下左右都添加一个0,[1,1,1,1] = [1,1,1,1,0,0,0,0]
pointwise_to_conv_weight = F.pad(conv_2d_pointwise.weight, [1,1,1,1])
pointwise_to_conv_weight.size() # 变成 [2,2,3,3] 改换kernel size = 3*3 的卷积了
# %%
conv_2d_for_pointwise = nn.Conv2d(in_channels, ou_channels, kernel_size, padding='same')
# 修改成padding后的权重
conv_2d_for_pointwise.weight = nn.Parameter(pointwise_to_conv_weight)
conv_2d_for_pointwise.bias = conv_2d_pointwise.bias
# %%
# x 写成卷积形式
# 全0矩阵
zeros = torch.unsqueeze(torch.zeros(kernel_size, kernel_size), 0)
zeros.size()

# %%
# 中间1矩阵
stars = torch.unsqueeze(F.pad(torch.ones(1,1), [1,1,1,1]), 0)
stars.size()
# %%
# 第一个输出通道卷积核
stars_zeros = torch.unsqueeze(torch.cat([stars, zeros],0),0)
stars_zeros.size()
# %%
# 第二个输出通道卷积核
zeros_stars = torch.unsqueeze(torch.cat([zeros, stars],0),0)
zeros_stars.size()

# %%
# 总的卷积核
identity_to_conv_weight = torch.cat([stars_zeros, zeros_stars], 0)
identity_to_conv_weight.size()
# %%
identity_to_conv_bias = torch.zeros([ou_channels])

# %%
# x 3*3卷积
conv_2d_for_identity = nn.Conv2d(in_channels, ou_channels, kernel_size, padding='same')
conv_2d_for_identity.weight = nn.Parameter(identity_to_conv_weight)
conv_2d_for_identity.bias = nn.Parameter(identity_to_conv_bias)
# %%
result2 = conv_2d(x) + conv_2d_for_pointwise(x) + conv_2d_for_identity(x)

# %%
print(torch.all(torch.isclose(result1, result2)))

# %%
# 融合
conv_2d_for_fusion = nn.Conv2d(in_channels, ou_channels, kernel_size, padding='same')
conv_2d_for_fusion.weight = nn.Parameter(conv_2d.weight.data 
                                            + conv_2d_for_pointwise.weight.data
                                            + conv_2d_for_identity.weight.data)
conv_2d_for_fusion.weight.size()
# %%
conv_2d_for_fusion.bias = nn.Parameter(conv_2d.bias.data 
                                            + conv_2d_for_pointwise.bias.data
                                            + conv_2d_for_identity.bias.data)
# %%
result3 = conv_2d_for_fusion(x)


# %%
print(torch.all(torch.isclose(result1,  result2)))
print(torch.all(torch.isclose(result1,  result3)))
print(torch.all(torch.isclose(result2,  result3)))
# %%
