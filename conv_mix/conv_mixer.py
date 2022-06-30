# %%
from re import sub
from pip import main
import torch
import torch.nn as nn
import torch.nn.functional as F
# %%

def convMixer(in_channel, h, depth, kernel_size, patch_size, n_classes, dropout=0.1):
    """convMixer 卷积混合模型
    h是输出通道
    depth是convMixer卷积深度
    """
    Seq = nn.Sequential
    ActBn = lambda x: Seq(x, nn.GELU(), nn.BatchNorm2d(h))
    Residual = type('Residual', (Seq,), {'forward':lambda self, x: self[0](x) + x} )
    return Seq(
            # nn.Conv2d(3, h, patch_size, stride=patch_size)即为patch embedding
            ActBn(nn.Conv2d(in_channel, h, patch_size, stride=patch_size)),
            nn.Dropout(0.1),
            # 搭建depth层convmixer layer, 
            *[Seq(
                # nn.Conv2d(h, h, kernel_size, groups=h, padding='same')即为空间融合，不考虑通道
                Residual(ActBn(nn.Conv2d(h, h, kernel_size, groups=h, padding='same'))),
                # nn.Conv2d(h, h, 1)即为point-wise卷积，不考虑向邻近的空间的点
                ActBn(nn.Conv2d(h, h, 1)),
                nn.Dropout(0.1)
                )
                for i in range(depth)
            ],
            # 平均池化层， 输出1*1
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(h, n_classes)
            )

# %%

class Net(nn.Module):
    def __init__(self,in_channel, h, depth, kernel_size, patch_size, n_classes):
        super(Net, self).__init__()
        self.conv_mixer = convMixer(in_channel, h, depth, kernel_size, patch_size, n_classes)
       
    def forward(self, x):
        x = self.conv_mixer(x)

        return F.log_softmax(x)
# %%
