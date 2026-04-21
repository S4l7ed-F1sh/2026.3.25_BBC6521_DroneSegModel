import torch
import torch.nn.functional as F
import numpy as np
import cv2
from matplotlib import pyplot as plt


def get_colors_for_visualization(n_classes=5):
    """
    生成指定数量的颜色

    Args:
        n_classes: 类别数量

    Returns:
        colors: RGB颜色列表，每个元素为(r, g, b)元组，值域0-255
    """
    # 使用gist_ncar颜色映射生成颜色
    cmap = plt.cm.get_cmap('gist_ncar', n_classes)
    colors = [(int(cmap(i)[0] * 255), int(cmap(i)[1] * 255), int(cmap(i)[2] * 255)) for i in range(n_classes)]
    return colors


def segmentation_visualizer(image, mask, edge_width=5, n_classes=5, method='contours'):
    """
    为分割结果生成可视化图像，内部使用半透明混合，边缘使用纯色

    Args:
        image: RGB图像张量，形状为(1, 3, H, W)，值域0-1或0-255
        mask: 分割mask张量，形状为(1, H, W)，值域为类别索引
        edge_width: 边缘宽度，默认为5
        n_classes: 类别数量，默认为5
        method: 可视化方法，这里固定为contours方式

    Returns:
        visualized_image: 可视化结果，形状为(H, W, 3)，值域0-255
    """
    # 确保batch size为1
    assert image.shape[0] == 1, f"Batch size must be 1, got {image.shape[0]}"
    assert mask.shape[0] == 1, f"Batch size must be 1, got {mask.shape[0]}"

    # 获取图像尺寸
    _, _, H, W = image.shape

    # 获取颜色
    colors = get_colors_for_visualization(n_classes)

    # 将图像转换为numpy数组
    img_np = image[0].permute(1, 2, 0).cpu().numpy()  # (H, W, 3)

    # 如果图像值域在[0,1]之间，转换到[0,255]
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    else:
        img_np = img_np.astype(np.uint8)

    # 将mask转换为numpy数组
    mask_np = mask[0].cpu().numpy().astype(np.int32)  # (H, W)

    # 创建输出图像副本
    output_img = img_np.copy().astype(np.float32)

    # 为每个类别单独处理
    unique_classes = np.unique(mask_np)

    for class_id in unique_classes:
        if class_id >= n_classes or class_id < 0:
            continue

        # 获取当前类别的mask
        class_mask = (mask_np == class_id)

        if not np.any(class_mask):
            continue

        # 获取当前类别的颜色
        color_rgb = colors[class_id]  # (r, g, b)

        # 获取当前类别的二值掩码
        class_mask_uint8 = class_mask.astype(np.uint8) * 255

        # 查找轮廓
        contours, _ = cv2.findContours(
            class_mask_uint8,
            cv2.RETR_EXTERNAL,  # 只检测外轮廓
            cv2.CHAIN_APPROX_NONE  # 存储所有轮廓点
        )

        # 创建当前类别的内部区域mask（排除边缘）
        inner_mask = np.zeros_like(class_mask_uint8)
        inner_mask[class_mask] = 255  # 先标记整个类别区域

        # 通过腐蚀获得内部区域（排除边缘）
        if edge_width > 0:
            kernel = np.ones((edge_width * 2 + 1, edge_width * 2 + 1), np.uint8)
            eroded = cv2.erode(class_mask_uint8, kernel, iterations=1)
            edge_mask = cv2.subtract(class_mask_uint8, eroded)  # 边缘区域
            inner_mask = eroded  # 内部区域
        else:
            edge_mask = np.zeros_like(class_mask_uint8)  # 没有边缘

        # 处理内部区域：0.5*原图 + 0.5*颜色
        inner_indices = (inner_mask > 0) & class_mask
        for c in range(3):  # RGB三通道
            output_img[inner_indices, c] = (
                    0.5 * img_np[inner_indices, c] +
                    0.5 * color_rgb[c]
            )

        # 处理边缘区域：使用纯色
        edge_indices = (edge_mask > 0) & class_mask
        for c in range(3):  # RGB三通道
            output_img[edge_indices, c] = color_rgb[c]

    return output_img.astype(np.uint8)