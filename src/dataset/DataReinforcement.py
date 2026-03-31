import numpy as np
import random


def data_augmentation(image, label, h_flip_prob=0.5, v_flip_prob=0.5,
                      stretch_range=(0.9, 1.1), color_jitter_prob=0.3):
    """
    语义分割数据增强函数（保持几何一致性）

    参数:
        image: np.array (H, W, 3) 原始图像
        label: np.array (H, W) 对应的标签图
        h_flip_prob: 水平翻转概率
        v_flip_prob: 垂直翻转概率
        stretch_range: 拉伸比例范围(最小值, 最大值)
        color_jitter_prob: 色彩抖动概率

    返回:
        augmented_image, augmented_label: 增强后的图像和标签
    """
    # 1. 水平翻转
    if random.random() < h_flip_prob:
        image = np.fliplr(image).copy()
        label = np.fliplr(label).copy()

    # 2. 垂直翻转
    if random.random() < v_flip_prob:
        image = np.flipud(image).copy()
        label = np.flipud(label).copy()

    # 3. 水平/垂直拉伸（保持几何一致性）
    if random.random() < 0.5:  # 50%概率进行拉伸
        h, w = image.shape[:2]
        # 随机选择水平或垂直拉伸
        if random.random() < 0.5:  # 水平拉伸
            scale = random.uniform(*stretch_range)
            new_w = int(w * scale)
            # 使用最近邻插值保持标签整数性
            image = np.array(Image.fromarray(image).resize((new_w, h),
                                                           Image.BICUBIC))
            label = np.array(Image.fromarray(label).resize((new_w, h),
                                                           Image.NEAREST))
        else:  # 垂直拉伸
            scale = random.uniform(*stretch_range)
            new_h = int(h * scale)
            image = np.array(Image.fromarray(image).resize((w, new_h),
                                                           Image.BICUBIC))
            label = np.array(Image.fromarray(label).resize((w, new_h),
                                                           Image.NEAREST))

        # 裁剪回原始尺寸（中心裁剪）
        h, w = image.shape[:2]
        start_x = max(0, (w - image.shape[1]) // 2)
        start_y = max(0, (h - image.shape[0]) // 2)
        image = image[start_y:start_y + image.shape[0], start_x:start_x + image.shape[1]]
        label = label[start_y:start_y + label.shape[0], start_x:start_x + label.shape[1]]

    # 4. 色彩抖动（仅对图像）
    if random.random() < color_jitter_prob:
        # 亮度调整
        brightness = random.uniform(0.8, 1.2)
        image = np.clip(image * brightness, 0, 255).astype(np.uint8)

        # 对比度调整
        contrast = random.uniform(0.8, 1.2)
        mean = np.mean(image, axis=(0, 1), keepdims=True)
        image = np.clip((image - mean) * contrast + mean, 0, 255).astype(np.uint8)

    # 5. 极弱高斯噪声（仅对图像）
    if random.random() < 0.2:  # 20%概率添加噪声
        noise = np.random.normal(0, 5, image.shape).astype(np.float32)
        image = np.clip(image + noise, 0, 255).astype(np.uint8)

    return image, label

# 使用示例
# augmented_img, augmented_lbl = data_augmentation(original_img, original_lbl)