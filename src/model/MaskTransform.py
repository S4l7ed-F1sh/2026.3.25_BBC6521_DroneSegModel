import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage


def multi_class_post_process(mask, permutation, minsize=50):
    """
    多类别分割mask后处理

    Args:
        mask: 输入mask张量，形状为(1, H, W)，值域为0-4
        permutation: 长度为5的列表，表示各类别的优先级顺序，[p0, p1, p2, p3, p4]，p0具有最高优先级
        minsize: 最小连通区域面积阈值，小于该值的区域将被移除

    Returns:
        processed_mask: 处理后的mask，形状为(1, H, W)，值域为0-4
    """
    assert mask.shape[0] == 1, f"Expected batch size 1, got {mask.shape[0]}"
    assert len(permutation) == 5, f"Expected permutation of length 5, got {len(permutation)}"

    batch_size, height, width = mask.shape
    device = mask.device
    dtype = mask.dtype

    # 第一步：提取每个类别的二分类mask
    binary_masks = []
    for i in range(5):
        binary_mask = (mask == i).float()  # (1, H, W)
        binary_masks.append(binary_mask)

    # 第二步：对每个二分类mask进行形态学处理和面积过滤
    processed_binary_masks = []
    for i in range(5):
        bin_mask = binary_masks[i][0]  # (H, W)

        # 转换为numpy进行连通组件分析
        mask_np = bin_mask.cpu().numpy()

        # 进行几次先膨胀后腐蚀的操作，去除小空洞
        # 使用kernel_size=3，进行2次闭运算
        kernel = torch.ones((3, 3), device=device, dtype=torch.float32)

        # 转换为tensor进行形态学操作
        temp_mask = bin_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        # 2次闭运算：先膨胀后腐蚀
        for _ in range(2):
            # 膨胀
            dilated = F.conv2d(temp_mask, kernel.unsqueeze(0).unsqueeze(0),
                               padding=1, groups=1)
            dilated = (dilated >= 1).float()

            # 腐蚀
            eroded = F.conv2d(dilated, kernel.unsqueeze(0).unsqueeze(0),
                              padding=1, groups=1)
            kernel_sum = kernel.sum()
            eroded = (eroded == kernel_sum).float()

            temp_mask = eroded

        # 转换回numpy进行连通组件分析
        processed_mask_np = temp_mask.squeeze().cpu().numpy()

        # 连通组件分析并移除小面积区域
        labeled, ncomponents = ndimage.label(processed_mask_np)
        component_sizes = ndimage.sum(processed_mask_np, labeled, range(ncomponents + 1))

        # 标记小面积组件
        small_components = component_sizes < minsize
        small_mask = small_components[labeled]

        # 移除小面积组件
        final_mask_np = processed_mask_np.copy()
        final_mask_np[small_mask] = 0

        # 转换回tensor
        final_mask_tensor = torch.from_numpy(final_mask_np).to(device=device, dtype=bin_mask.dtype)
        processed_binary_masks.append(final_mask_tensor)

    # 第三步：按照优先级顺序组合mask
    # 初始化输出mask为最低优先级类别（p4）
    p4 = permutation[4]
    output_mask = torch.full((height, width), fill_value=p4, device=device, dtype=dtype)

    # 按优先级从高到低覆盖：p3覆盖p4，p2覆盖p3，p1覆盖p2，p0覆盖p1
    for priority_idx in range(3, -1, -1):  # 从3到0
        class_val = permutation[priority_idx]  # 当前优先级的类别值
        class_bin_mask = processed_binary_masks[class_val]  # 获取该类别的二分类mask

        # 使用该类别的mask覆盖相应区域
        output_mask = torch.where(class_bin_mask.bool(), class_val, output_mask)

    return output_mask.unsqueeze(0)  # 返回形状为(1, H, W)