import numpy as np
import random
from PIL import Image


def data_augmentation(image, label, h_flip_prob=0.5, v_flip_prob=0.5,
                      stretch_factor=1.2, color_jitter_prob=0.3):
    """
    语义分割数据增强函数（严格保持输出尺寸一致）

    修改说明：
    - 移除了收缩操作，只允许放大后中心裁剪，确保输出分辨率不变。
    """
    # 1. 类型安全检查
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    if label.dtype != np.uint8:
        label = label.astype(np.uint8)

    # 记录原始尺寸，用于最后恢复
    orig_h, orig_w = image.shape[:2]

    # 2. 水平翻转
    if random.random() < h_flip_prob:
        image = np.fliplr(image).copy()
        label = np.fliplr(label).copy()

    # 3. 垂直翻转
    if random.random() < v_flip_prob:
        image = np.flipud(image).copy()
        label = np.flipud(label).copy()

    # 4. 随机拉伸（只放大，后接中心裁剪）
    if random.random() < 0.5:
        scale = random.uniform(1.0, stretch_factor)  # 只生成 >= 1.0 的缩放因子

        # 随机选择水平或垂直拉伸
        if random.random() < 0.5:  # 水平拉伸
            new_w = int(orig_w * scale)
            target_size = (new_w, orig_h)  # PIL resize 需要 (W, H)

            image = np.array(Image.fromarray(image).resize(target_size, Image.BICUBIC))
            label = np.array(Image.fromarray(label).resize(target_size, Image.NEAREST))
        else:  # 垂直拉伸
            new_h = int(orig_h * scale)
            target_size = (orig_w, new_h)

            image = np.array(Image.fromarray(image).resize(target_size, Image.BICUBIC))
            label = np.array(Image.fromarray(label).resize(target_size, Image.NEAREST))

        # --- 关键修正：中心裁剪回原始尺寸 ---
        cur_h, cur_w = image.shape[:2]

        # 计算中心裁剪的起始点
        start_x = (cur_w - orig_w) // 2
        start_y = (cur_h - orig_h) // 2

        # 执行裁剪，确保输出尺寸严格等于 orig_h, orig_w
        image = image[start_y:start_y + orig_h, start_x:start_x + orig_w]
        label = label[start_y:start_y + orig_h, start_x:start_x + orig_w]

    # 5. 色彩抖动
    if random.random() < color_jitter_prob:
        brightness = random.uniform(0.8, 1.2)
        image = np.clip(image * brightness, 0, 255).astype(np.uint8)

        contrast = random.uniform(0.8, 1.2)
        mean_val = np.mean(image, axis=(0, 1), keepdims=True)
        image = np.clip((image - mean_val) * contrast + mean_val, 0, 255).astype(np.uint8)

    # 6. 极弱高斯噪声
    if random.random() < 0.2:
        noise = np.random.normal(0, 5, image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return image, label