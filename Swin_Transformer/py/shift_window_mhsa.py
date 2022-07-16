import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import import_ipynb


def window2image(msa_output):
    """
    输入：[bs, num_window, num_patch_in_window, patch_depth]
    输出：[bs, c, image_h, image_w]
    其中：
            c = patch_depth
            image_h = int(math.sqrt(num_window)) * int(math.sqrt(num_patch_in_window))
            image_w = image_h
    """
    bs, num_window, num_patch_in_window, patch_depth = msa_output.shape
    window_size = int(math.sqrt(num_patch_in_window))
    image_h = image_w = int(math.sqrt(num_window)) * window_size

    msa_output = msa_output.reshape([
        bs,
        int(math.sqrt(num_window)),
        int(math.sqrt(num_window)),
        window_size,  # window_size * window_size = num_patch_in_window
        window_size,
        patch_depth
    ])
    msa_output = msa_output.transpose(2, 3)

    # 转化为三维格式
    image = msa_output.reshape([bs, image_h * image_w, patch_depth])
    image = image.transpose(-1, -2)
    image = image.reshape([bs, patch_depth, image_h, image_w])

    return image


def build_mask_for_shifted_window_mhsa(
            bs,
            image_h,
            image_w,
            window_size
        ):
    """
    1.  首先构建—个shift-window的patch所属的window类别矩阵
        类别矩阵 index_matrix：
            1 2 2 3
            4 5 5 6
            4 5 5 6
            7 8 8 9
        类别矩阵含有9个window区域

    2.  对类别矩阵进行往左和往上各自滑动半个窗口大小的步长的操作

    3. 通过unfold操作得到[bs, num_window, num_patch_in_window]形状的类别矩阵

    4.  对该矩阵进行扩维成[bs, num_window, num_patch_in_window,1]的4维张量

    5. 将该矩阵与其转置矩阵进行作差，a - a^T，得到同类关系矩阵（元素为0的位置上的patch属于同类，
    否则属于不同类)

    对同类关系矩阵中非零的位置用负无穷数进行填充，对于零的位置用0去填充，
    这样就构建好了MHSA所需要的masK

    output:[bs*num_window,
            num_patch_in_window,
            num_patch_in_window]
    """
    index_matrix = torch.zeros(image_h, image_w)

    for i in range(image_h):
        for j in range(image_w):
            # 按window_size区块划分行的序号
            row_times = (i + window_size//2) // window_size
            # 按window_size区块划分列的序号
            col_times = (j + window_size//2) // window_size
            # row_times*(image_h // window_size) 上面若干行总共经过了多少区块
            index_matrix[i, j] = row_times*(image_h // window_size) + col_times + row_times + 1

    # print(index_matrix, "# index_matrix")  # 调试代码

    # 让类别矩阵向左向上滑动半个窗程
    rolled_index_matrix = torch.roll(index_matrix,
                                     shifts=(-window_size//2, -window_size//2),
                                     dims=(0, 1))
    # print(rolled_index_matrix, "# rolled_index_matrix")  # 调试代码

    # 引入bs和channel维度 [bs, ch, h, w]
    rolled_index_matrix = rolled_index_matrix.unsqueeze(0).unsqueeze(0)

    # 按照标准形式去划分窗口 c: [bs, num_window, num_patch_in_window]
    unfold_rolled_index_matrix = \
        F.unfold(rolled_index_matrix, kernel_size=window_size,
                 stride=window_size).transpose(-1, -2).tile(bs, 1, 1)

    # print("unfold_rolled_index_matrix: \n", unfold_rolled_index_matrix)  # 调试代码

    bs, num_window, num_patch_in_window = unfold_rolled_index_matrix.shape
    # 扩一维 c:[bs, num_window, num_patch_in_window,1]
    c1 = unfold_rolled_index_matrix.unsqueeze(-1)

    valid_matrix = ((c1 - c1.transpose(-1, -2)) == 0).to(torch.float32)
    # 不属于同一个窗口的转化成负无穷
    additive_mask = (1 - valid_matrix) * (-1e9)

    # print(additive_mask, "# additive_mask")  # 调试代码

    additive_mask = additive_mask.reshape(bs*num_window,
                                          num_patch_in_window,
                                          num_patch_in_window)

    return additive_mask


def shift_window(
        w_msa_output,
        window_size,
        shift_size,
        generate_mask=True):
    """辅助shift window函数，高效计算sw msa
    输入：w_mhsa_output: [bs, num_window, num_patch_in_window, patch_depth]
    输出：shift_window_output:[ bs, num_window,
                               num_patch_in_window, patch_depth]

         mask: [ bs*num_window,
                 num_patch_in_window,
                 num_patch_in_window] if generate_mask==True
    """
    bs, num_window, num_patch_in_window, patch_depth = w_msa_output.shape
    # 转换为image形式
    w_msa_output = window2image(w_msa_output)
    bs, patch_depth, image_h, image_w = w_msa_output.shape

    # 把图片的h,w维度往左和往上滑动半个窗程
    rolled_w_msa_output = torch.roll(
        w_msa_output,
        shifts=(shift_size, shift_size),
        dims=(2, 3)
    )
    # 再把shift后图片还原成patch格式
    shifted_w_msa_input = rolled_w_msa_output.reshape([
        bs,
        patch_depth,
        int(math.sqrt(num_window)),
        window_size,
        int(math.sqrt(num_window)),
        window_size,
    ])

    shifted_w_msa_input = shifted_w_msa_input.transpose(3, 4)

    shifted_w_msa_input = shifted_w_msa_input.reshape([
        bs,
        patch_depth,
        num_window * num_patch_in_window
    ])

    shifted_w_msa_input = shifted_w_msa_input.transpose(-1, -2)

    shifted_window = shifted_w_msa_input.reshape([
        bs,
        num_window,
        num_patch_in_window,
        patch_depth
    ])

    if generate_mask:
        additive_mask = build_mask_for_shifted_window_mhsa(
            bs,
            image_h,
            image_w,
            window_size
        )
    else:
        additive_mask = None

    return shifted_window, additive_mask


def shift_window_multi_head_self_attention(
        w_msa_output,
        mhsa,
        window_size,
):
    bs, num_window, num_patch_in_window, patch_depth = w_msa_output.shape
    # 对patch进行shift，向左上角滑动
    shifted_w_msa_input, additive_mask = shift_window(
        w_msa_output,
        window_size,
        shift_size=-window_size//2,
        generate_mask=True
    )
    # 转换成mhsa所需要的格式 [bs, seq_len, dim]
    shifted_w_msa_input = shifted_w_msa_input.reshape([bs*num_window,
                                                       num_patch_in_window,
                                                       patch_depth])
    # 计算shift mhsa, output: [bs, seq_len, dim]
    prob, output = mhsa(shifted_w_msa_input,
                        additive_mask=additive_mask)

    output = output.reshape([bs, num_window, num_patch_in_window, patch_depth])

    # 最后反shift一下，向右下角滑动
    output, _ = shift_window(output, window_size, shift_size=window_size//2,
                             generate_mask=False)
    return output
