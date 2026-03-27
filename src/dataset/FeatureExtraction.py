import numpy as np
import cv2
import torch

def extract_features(image):
    """
    对输入的RGB图像进行多维度特征提取，并将输出转换为特定格式。

    Args:
        image (np.ndarray): 输入图像，形状为 (H, W, 3)，数据类型为 uint8 (0~255)。

    Returns:
        torch.Tensor: 形状为 (1, 22, H, W) 的张量，包含原图、灰度图、Sobel导数及量化特征。
    """
    # 确保输入是float64以避免计算中的整数溢出问题
    img_f = image.astype(np.float64)

    # --- 1. 计算灰度图 (gx) ---
    gx = np.dot(img_f[..., :3], [0.299, 0.587, 0.114])
    gx = gx.astype(np.uint8)

    # --- 2. 计算Sobel偏导数 (dx) ---
    gx_f = gx.astype(np.float64)
    sobel_x1 = cv2.Sobel(gx_f, cv2.CV_64F, 1, 0, ksize=5)  # x方向一阶导
    sobel_y1 = cv2.Sobel(gx_f, cv2.CV_64F, 0, 1, ksize=5)  # y方向一阶导
    sobel_x2 = cv2.Sobel(gx_f, cv2.CV_64F, 2, 0, ksize=5)  # x方向二阶导
    sobel_y2 = cv2.Sobel(gx_f, cv2.CV_64F, 0, 2, ksize=5)  # y方向二阶导
    sobel_x3 = cv2.Sobel(gx_f, cv2.CV_64F, 3, 0, ksize=5)  # x方向三阶导
    sobel_y3 = cv2.Sobel(gx_f, cv2.CV_64F, 0, 3, ksize=5)  # y方向三阶导

    # 对sobel结果进行处理
    dx = np.stack([sobel_x1, sobel_y1, sobel_x2, sobel_y2, sobel_x3, sobel_y3], axis=-1)
    dx = dx + 128  # 偏移
    dx = np.clip(dx, 0, 255)  # 截断
    dx = dx / 255.0  # 归一化到[0, 1]

    # --- 3. 定义量化函数 ---
    def quantize_image(img_uint8, num_levels):
        step_size = 256 / num_levels
        quantized = (img_uint8 // step_size).astype(np.float32) / (num_levels - 1)  # 归一化
        return quantized

    combined_for_quantization = np.concatenate(
        [image, gx[..., np.newaxis]], axis=-1
    ).astype(np.uint8)

    qx1 = quantize_image(combined_for_quantization, 32)  # 32-quantized -> (H, W, 4)
    qx2 = quantize_image(combined_for_quantization, 16)  # 16-quantized -> (H, W, 4)
    qx3 = quantize_image(combined_for_quantization, 8)  # 8-quantized  -> (H, W, 4)

    # --- 4. 归一化和转换为浮点数 ---
    image_normalized = image.astype(np.float32) / 255.0
    gx_normalized = gx.astype(np.float32) / 255.0

    # --- 5. 将所有特征在通道维度上堆叠 ---
    final_features = np.concatenate([
        image_normalized,
        gx_normalized[..., np.newaxis],
        dx,
        qx1,
        qx2,
        qx3
    ], axis=-1)

    # 调整维度顺序并转为tensor
    final_features = np.transpose(final_features, (2, 0, 1))  # 转换为(C, H, W)
    final_features_tensor = torch.tensor(final_features, dtype=torch.float32)  # 添加批次维度

    # final_features_tensor = torch.tensor(final_features, dtype=torch.float32).unsqueeze(0)  # 添加批次维度
    # assert final_features_tensor.shape[1] == 22, f"最终特征通道数应为22，但实际为{final_features_tensor.shape[1]}"

    return final_features_tensor


# 示例用法
if __name__ == "__main__":
    example_image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    print(f"输入图像形状: {example_image.shape}")

    features = extract_features(example_image)
    print(f"输出特征形状: {features.shape}")
