import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage


def morphological_closing(mask_tensor, kernel_size=3, min_area=50):
    """
    对二分类mask执行闭运算（先膨胀后腐蚀）并去除小面积噪声

    Args:
        mask_tensor: 3维tensor，形状为(batch_size, H, W)，值域为0和1
        kernel_size: 形态学操作使用的核大小，默认为3
        min_area: 最小连通区域面积阈值，小于该值的区域将被移除

    Returns:
        processed_masks: 处理后的mask，形状与输入相同
    """
    if len(mask_tensor.shape) != 3:
        raise ValueError(f"Expected 3D tensor (batch_size, H, W), got {mask_tensor.shape}")

    batch_size, height, width = mask_tensor.shape
    device = mask_tensor.device
    dtype = mask_tensor.dtype

    # 创建方形核
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=device, dtype=torch.float32)

    processed_masks = []

    for i in range(batch_size):
        current_mask = mask_tensor[i].unsqueeze(0).unsqueeze(0).float()

        # 1. 膨胀操作
        dilated = F.conv2d(current_mask, kernel, padding=kernel_size // 2, groups=1)
        dilated = (dilated > 0).float()

        # 2. 腐蚀操作
        # 腐蚀意味着只有当核覆盖的所有像素都是前景时，中心像素才是前景
        eroded = F.conv2d(dilated, kernel, padding=kernel_size // 2, groups=1)
        kernel_total = kernel.sum()
        eroded = (eroded == kernel_total).float()

        # 3. 连通组件分析并移除小区域
        mask_np = eroded.squeeze().cpu().numpy()

        # 找连通组件
        labeled, ncomponents = ndimage.label(mask_np)

        # 计算每个组件的面积
        areas = ndimage.sum(mask_np, labeled, range(ncomponents + 1))

        # 创建mask来标识面积过小的组件
        small_components = areas < min_area
        small_mask = small_components[labeled]

        # 移除小面积组件
        cleaned_mask = mask_np.copy()
        cleaned_mask[small_mask] = 0

        # 转换回tensor
        result = torch.from_numpy(cleaned_mask).to(device=device, dtype=dtype)
        processed_masks.append(result)

    return torch.stack(processed_masks, dim=0)


def process_multi_class_segmentation_mask(masks, permutation, kernel_size=3, min_area=50):
    """
    对多类别分割mask进行后处理，按优先级重新组合

    Args:
        masks: 3维tensor，形状为(batch_size, H, W)，值域为0, 1, 2, 3, 4
        permutation: 长度为5的列表，表示各类别的优先级顺序，越靠前优先级越高
        kernel_size: 形态学操作使用的核大小，默认为3
        min_area: 最小连通区域面积阈值，小于该值的区域将被移除

    Returns:
        processed_masks: 处理后的mask，形状与输入相同，值域为0, 1, 2, 3, 4
    """
    if len(masks.shape) != 3:
        raise ValueError(f"Expected 3D tensor (batch_size, H, W), got {masks.shape}")

    if len(permutation) != 5:
        raise ValueError(f"Permutation should have length 5, got {len(permutation)}")

    batch_size, height, width = masks.shape
    device = masks.device

    # 将输入转换为float类型以便后续处理
    masks_float = masks.float()

    # 分解为5个二分类mask
    binary_masks = []
    for class_idx in range(5):
        binary_mask = (masks_float == float(class_idx)).float()
        binary_masks.append(binary_mask)

    # 将binary_masks堆叠为(batch_size, 5, H, W)
    binary_masks = torch.stack(binary_masks, dim=1)  # (batch_size, 5, H, W)

    # 对每个类别分别进行形态学处理
    processed_binary_masks = []
    for class_idx in range(5):
        class_masks = binary_masks[:, class_idx, :, :]  # (batch_size, H, W)
        processed_class_masks = morphological_closing(class_masks, kernel_size, min_area)
        processed_binary_masks.append(processed_class_masks)

    # 重新堆叠处理后的二分类mask
    processed_binary_masks = torch.stack(processed_binary_masks, dim=1)  # (batch_size, 5, H, W)

    # 根据permutation重新组织优先级
    # 创建输出mask，初始化为最低优先级的类别
    output_masks = torch.zeros_like(masks, dtype=torch.long)

    # 按照permutation中的优先级顺序，从高到低依次填充
    # permutation中越靠前的元素优先级越高
    for priority_idx, class_value in enumerate(permutation):
        class_binary_mask = processed_binary_masks[:, class_value, :, :]  # (batch_size, H, W)

        # 只有在当前位置还没有被更高优先级的类别占据时，才设置当前类别
        # 由于我们按优先级顺序处理，所以可以直接更新
        mask_to_update = class_binary_mask.bool()
        output_masks[mask_to_update] = class_value

    return output_masks


def multi_class_post_process(masks, permutation, kernel_size=3, min_area=50):
    """
    多类别分割mask后处理的主函数

    Args:
        masks: 3维tensor，形状为(batch_size, H, W)，值域为0, 1, 2, 3, 4，类型为long
        permutation: 长度为5的列表，表示各类别的优先级顺序，越靠前优先级越高
        kernel_size: 形态学操作使用的核大小，默认为3
        min_area: 最小连通区域面积阈值，小于该值的区域将被移除

    Returns:
        processed_masks: 处理后的mask，形状与输入相同，值域为0, 1, 2, 3, 4
    """
    return process_multi_class_segmentation_mask(masks, permutation, kernel_size, min_area)