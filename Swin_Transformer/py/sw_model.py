import torch
import torch.nn as nn
import torch.nn.functional as F
import py.image2emb as image2emb
import py.patch_merge as patch_merge
import py.sw_block as sw_block


class SwinTransformerModel(nn.Module):
    """
    输入：image格式 [bs, c, h, w]
    输出：[bs, num_class]

        1.首先对图片进行分块并得到Patch embedding

        2.经过第一个stage(这里只有一个block)

        3.进行patch merging，再进行第二个stage，以此循环往下...

        4.对最后一个block的输出转换成patch embedding的格式,[bs, num_patch, patch_depth]

        5.对patch embedding在num_patch维度（时间维度）进行平均池化，
          并映射到分类层得到分类的logits
    """
    def __init__(self, input_image_channel=3, patch_size=4, model_dim_c=8, num_class=10, window_size=4, num_head=2,
                 merge_size=2):
        super(SwinTransformerModel, self).__init__()
        self.merge_size = merge_size
        self.num_head = num_head
        self.window_size = window_size
        self.num_class = num_class
        self.patch_size = patch_size
        self.input_image_channel = input_image_channel
        self.model_dim_C = model_dim_c

        patch_depth = patch_size * patch_size * input_image_channel
        # weight定义成nn.Parameter格式参与到梯度更新
        self.patch_embedding_weight = nn.Parameter(torch.randn(patch_depth, model_dim_c))
        self.block1 = sw_block.SwinTransformerBlock(model_dim_c, window_size, num_head)
        self.block2 = sw_block.SwinTransformerBlock(model_dim_c*2, window_size, num_head)
        self.block3 = sw_block.SwinTransformerBlock(model_dim_c*4, window_size, num_head)
        self.block4 = sw_block.SwinTransformerBlock(model_dim_c*8, window_size, num_head)

        self.patch_merging1 = patch_merge.PatchMerging(model_dim_c, merge_size)
        self.patch_merging2 = patch_merge.PatchMerging(model_dim_c*2, merge_size)
        self.patch_merging3 = patch_merge.PatchMerging(model_dim_c*4, merge_size)

        # MLP映射到分类
        self.final_layer = nn.Linear(model_dim_c*8, num_class)

    def forward(self, image):
        patch_embedding = image2emb.image2emb_naive(image,
                                                    self.patch_size,
                                                    self.patch_embedding_weight)
        print(patch_embedding.shape, "# patch_emb.shape")

        # block1 + merge1(patch缩小1/4, dim扩大2倍，下merge同)
        sw_mhsa_output1 = self.block1(patch_embedding)
        merged_patch1 = self.patch_merging1(sw_mhsa_output1)
        print(sw_mhsa_output1.shape, "# sw_mhsa_output1.shape")
        print(merged_patch1.shape, "# merged_patch1.shape")

        # block2 + merge2
        sw_mhsa_output2 = self.block2(merged_patch1)
        merged_patch2 = self.patch_merging2(sw_mhsa_output2)
        print(sw_mhsa_output2.shape, "# sw_mhsa_output2.shape")
        print(merged_patch2.shape, "# merged_patch2.shape")

        # block3 + merge3
        sw_mhsa_output3 = self.block3(merged_patch2)
        merged_patch3 = self.patch_merging3(sw_mhsa_output3)
        print(sw_mhsa_output3.shape, "# sw_mhsa_output3.shape")
        print(merged_patch3.shape, "# merged_patch3.shape")

        # block4
        sw_mhsa_output4 = self.block4(merged_patch3)
        print(sw_mhsa_output4.shape, "# sw_mhsa_output4.shape")

        # sw_mhsa_output4：window格式
        bs, num_window, num_patch_in_window, patch_depth = sw_mhsa_output4.shape
        # 转化为3维的
        sw_mhsa_output3 = sw_mhsa_output3.reshape([bs, -1, patch_depth])
        # 平均池化时间维度
        pool_output = torch.mean(sw_mhsa_output3, dim=1)  # [bs, patch_depth]

        logits = self.final_layer(pool_output)
        print(logits.shape, "# logits.shape")

        return logits


def test():
    bs, ic, h, w = 2, 3, 256, 256
    patch_size = 4  # 4×4作为一个patch
    model_dim = 8
    num_class = 10
    window_size = 4  # 一个窗有4×4个patch
    num_head = 2
    merge_size = 2  # 论文实现为2
    image = torch.randn([bs, ic, h, w])
    model = SwinTransformerModel(input_image_channel=ic,
                                 patch_size=patch_size,
                                 model_dim_c=model_dim,
                                 num_class=num_class,
                                 window_size=window_size,
                                 num_head=num_head,
                                 merge_size=merge_size)
    logits = model(image)
    output = F.softmax(logits[0], dim=0)
    print(output)


if __name__ == '__main__':
    test()



