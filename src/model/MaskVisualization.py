import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2


def get_colors_for_visualization(n_classes=5):
    """
    使用 gist_ncar 颜色映射生成指定数量的颜色

    Args:
        n_classes: 类别数量，默认为5

    Returns:
        colors_float: RGB颜色列表，每个元素为(r, g, b)元组，值域0-1
        color_list_int: 整数颜色列表，格式[r, g, b, r, g, b, ...]，值域0-255
    """
    # 使用gist_ncar颜色映射
    cmap = plt.cm.get_cmap('gist_ncar', n_classes)

    # 获取RGBA列表，并只取前3个通道(RGB)
    colors_float = [cmap(i)[:3] for i in range(n_classes)]

    # 转换为0-255整数，并展平列表
    color_list_int = (np.array(colors_float) * 255).astype(int).flatten().tolist()

    # 补全调色板(PIL P模式需要768个值)
    if len(color_list_int) < 768:
        color_list_int += [0] * (768 - len(color_list_int))

    return colors_float, color_list_int


def visualize_segmentation_with_edges(image, mask, edge_width=2, n_classes=5):
    """
    为分割结果生成可视化图像，包含边缘高亮效果

    Args:
        image: RGB图像张量，形状为(1, 3, H, W)，值域0-1或0-255
        mask: 分割mask张量，形状为(1, H, W)，值域为类别索引
        edge_width: 边缘宽度，默认为2
        n_classes: 类别数量，默认为5

    Returns:
        visualized_image: 可视化结果，形状为(H, W, 3)，值域0-255
    """
    # 确保batch size为1
    assert image.shape[0] == 1, f"Batch size must be 1, got {image.shape[0]}"
    assert mask.shape[0] == 1, f"Batch size must be 1, got {mask.shape[0]}"

    # 获取图像尺寸
    _, _, H, W = image.shape

    # 获取颜色
    colors_float, _ = get_colors_for_visualization(n_classes)

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
    output_img = img_np.copy()

    # 为每个类别单独处理
    unique_classes = np.unique(mask_np)

    for class_id in unique_classes:
        if class_id >= n_classes:
            continue

        # 获取当前类别的mask
        class_mask = (mask_np == class_id)

        if not np.any(class_mask):
            continue

        # 获取当前类别的颜色
        color_rgb = tuple((np.array(colors_float[class_id]) * 255).astype(int))

        # 计算边缘（通过膨胀和腐蚀计算边界）
        class_mask_uint8 = class_mask.astype(np.uint8) * 255

        # 使用OpenCV计算边缘
        kernel = np.ones((3, 3), np.uint8)

        # 腐蚀
        eroded = cv2.erode(class_mask_uint8, kernel, iterations=1)
        # 边缘 = 原始 - 腐蚀结果
        edges = cv2.morphologyEx(class_mask_uint8 - eroded, cv2.MORPH_CLOSE,
                                 np.ones((edge_width, edge_width), np.uint8))

        # 扩大边缘区域以达到指定的边缘宽度
        if edge_width > 1:
            expanded_kernel = np.ones((edge_width * 2 + 1, edge_width * 2 + 1), np.uint8)
            edges = cv2.dilate(edges, expanded_kernel, iterations=1)
            # 确保边缘不超出原始mask范围
            edges = np.logical_and(edges.astype(bool), class_mask)

        # 获取内部区域（非边缘部分）
        inner_region = np.logical_and(class_mask, ~edges.astype(bool))

        # 绘制边缘：纯色
        output_img[edges.astype(bool)] = color_rgb

        # 绘制内部区域：半透明混合（一半原图+一半颜色）
        for c in range(3):  # 对RGB三个通道分别处理
            output_img[inner_region, c] = (
                    0.5 * img_np[inner_region, c] +
                    0.5 * color_rgb[c]
            ).astype(np.uint8)

    return output_img


def segmentation_visualizer(image, mask, edge_width=10, n_classes=5):
    """
    分割结果可视化主函数

    Args:
        image: RGB图像张量，形状为(1, 3, H, W)，值域0-1或0-255
        mask: 分割mask张量，形状为(1, H, W)，值域为类别索引
        edge_width: 边缘宽度，默认为10
        n_classes: 类别数量，默认为5

    Returns:
        visualized_image: 可视化结果，形状为(H, W, 3)，值域0-255
    """
    return visualize_segmentation_with_edges(image, mask, edge_width, n_classes)


def visualize_with_contours(image, mask, edge_width=10, n_classes=5):
    """
    使用轮廓检测方法生成可视化图像

    Args:
        image: RGB图像张量，形状为(1, 3, H, W)，值域0-1或0-255
        mask: 分割mask张量，形状为(1, H, W)，值域为类别索引
        edge_width: 边缘宽度，默认为10
        n_classes: 类别数量，默认为5

    Returns:
        visualized_image: 可视化结果，形状为(H, W, 3)，值域0-255
    """
    # 确保batch size为1
    assert image.shape[0] == 1, f"Batch size must be 1, got {image.shape[0]}"
    assert mask.shape[0] == 1, f"Batch size must be 1, got {mask.shape[0]}"

    # 获取图像尺寸
    _, _, H, W = image.shape

    # 获取颜色
    colors_float, _ = get_colors_for_visualization(n_classes)

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
    output_img = img_np.copy()

    # 为每个类别单独处理
    unique_classes = np.unique(mask_np)

    for class_id in unique_classes:
        if class_id >= n_classes or class_id < 0:
            continue

        # 获取当前类别的mask
        class_mask = (mask_np == class_id).astype(np.uint8)

        if not np.any(class_mask):
            continue

        # 获取当前类别的颜色
        color_rgb = tuple((np.array(colors_float[class_id]) * 255).astype(int))

        # 查找轮廓
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 在输出图像上绘制半透明填充区域
        overlay = output_img.copy()
        for c in range(3):  # 对RGB三个通道分别处理半透明效果
            overlay[class_mask.astype(bool), c] = (
                    0.5 * img_np[class_mask.astype(bool), c] +
                    0.5 * color_rgb[c]
            ).astype(np.uint8)

        # 将overlay与output_img混合
        alpha = 0.5
        output_img = (alpha * output_img + (1 - alpha) * overlay).astype(np.uint8)

        # 绘制边缘轮廓
        cv2.drawContours(output_img, contours, -1, color_rgb, thickness=edge_width)

    return output_img