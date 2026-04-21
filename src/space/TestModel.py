from PIL import Image
import numpy as np
import torch
from src.dataset.FeatureExtraction import extract_features
import sys
import matplotlib.pyplot as plt

from ModelLoadAndWork import load_model, run_inference


def visualize_original_mask(mask, title="Original Mask"):
    """
    可视化原始mask，使用不同颜色区分不同类别

    Args:
        mask: 形状为(1, H, W)或(H, W)的tensor或numpy array
        title: 图像标题
    """
    # 确保mask是numpy array
    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = mask

    # 如果是batch维度，去掉它
    if mask_np.ndim == 3:
        mask_np = mask_np[0]

    # 获取类别数量
    unique_values = np.unique(mask_np)
    print(f"{title} 中的类别: {unique_values}")
    print(f"{title} 中各类别的像素数量: {np.bincount(mask_np.flatten())}")

    # 创建彩色mask
    h, w = mask_np.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

    # 定义颜色映射
    colors = [
        [0, 0, 0],  # 背景 - 黑色
        [255, 0, 0],  # 类别1 - 红色
        [0, 255, 0],  # 类别2 - 绿色
        [0, 0, 255],  # 类别3 - 蓝色
        [255, 255, 0],  # 类别4 - 黄色
        [255, 0, 255],  # 类别5 - 紫色
        [0, 255, 255],  # 类别6 - 青色
        [192, 192, 192]  # 类别7 - 灰色
    ]

    # 为每个类别分配颜色
    for class_id in unique_values:
        if class_id < len(colors):
            colored_mask[mask_np == class_id] = colors[int(class_id)]
        else:
            # 如果类别超过预定义颜色数量，使用随机颜色
            random_color = np.random.randint(0, 256, 3)
            colored_mask[mask_np == class_id] = random_color

    # 显示图像
    plt.figure(figsize=(10, 8))
    plt.imshow(colored_mask)
    plt.title(title)
    plt.axis('off')
    plt.colorbar(label='Class ID')
    plt.show()


def visualize_prediction_heatmap(output, class_idx, title=None):
    """
    可视化特定类别的预测热力图

    Args:
        output: 模型输出，形状为(1, num_classes, H, W)
        class_idx: 要可视化的类别索引
        title: 图像标题
    """
    if title is None:
        title = f"Prediction Heatmap for Class {class_idx}"

    # 获取特定类别的预测分数
    pred_scores = output[0, class_idx, :, :].cpu().numpy()

    # 可视化热力图
    plt.figure(figsize=(10, 8))
    plt.imshow(pred_scores, cmap='hot', interpolation='nearest')
    plt.title(title)
    plt.colorbar(label='Prediction Score')
    plt.axis('off')
    plt.show()

    # 打印统计信息
    print(f"Class {class_idx} - Max score: {pred_scores.max():.3f}, "
          f"Min score: {pred_scores.min():.3f}, "
          f"Mean score: {pred_scores.mean():.3f}")


if __name__ == "__main__":
    img_path = './150.png'

    # 加载模型并获取模型实例
    model = load_model()  # 注意：这应该返回模型实例而不是无返回值
    if model is None:
        print("Error: Failed to load model!")
        sys.exit(1)

    image = Image.open(img_path)

    # 将图像转换为RGB格式，再转换为tensor
    image = image.convert('RGB')
    image = np.array(image)  # 转换为numpy数组 (H, W, 3)
    # image = torch.from_numpy(image).permute(2, 0, 1).float()

    print(f"输入图像形状: {image.shape}, 类型: {type(image)}")

    # 确保输入图像是正确的格式 (H, W, 3) 并且是 float32
    if image.dtype != np.float32:
        image = image.astype(np.float32)

    # 转换为 PyTorch 张量 (H, W, 3) -> (3, H, W)
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()

    # 提取特征 (22, H, W)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_tensor = extract_features(image).to(device)

    print("输入图像张量形状:", image_tensor.shape, "数据类型:", image_tensor.dtype)
    print("模型输入张量形状:", input_tensor.shape, "数据类型:", input_tensor.dtype)
    sys.stdout.flush()

    with torch.no_grad():
        output = model(input_tensor.unsqueeze(0))  # [1, 5, H, W]

    mask = torch.argmax(output, dim=1)  # [1, H, W]

    # 可视化原始mask
    print("\n=== 检查原始mask ===")
    visualize_original_mask(mask, "Original Predicted Mask")

    # 打印详细统计信息
    mask_np = mask.cpu().numpy()[0]  # 去掉batch维度
    unique_vals, counts = np.unique(mask_np, return_counts=True)
    print(f"\n原始mask中的类别分布:")
    for val, count in zip(unique_vals, counts):
        percentage = (count / mask_np.size) * 100
        print(f"  类别 {val}: {count} 个像素 ({percentage:.2f}%)")

    # 可视化每个类别的预测热力图
    print("\n=== 可视化各类别预测热力图 ===")
    for i in range(5):
        visualize_prediction_heatmap(output, i, f"Prediction Heatmap for Class {i}")

    # 检查是否存在某个类别的预测分数非常低
    print(f"\n=== 各类别最大预测分数 ===")
    for i in range(5):
        max_score = output[0, i, :, :].max().item()
        mean_score = output[0, i, :, :].mean().item()
        print(f"  类别 {i}: 最大分数 = {max_score:.3f}, 平均分数 = {mean_score:.3f}")

    # 生成二进制mask
    class_mask = torch.stack([
        (mask == i).float() for i in range(5)
    ], dim=1)  # [1, 5, H, W]

    print("\n模型输出张量形状:", output.shape, "数据类型:", output.dtype)
    print("模型输出张量形状:", mask.shape, "数据类型:", mask.dtype)
    print("模型输出张量形状:", class_mask.shape, "数据类型:", class_mask.dtype)
    sys.stdout.flush()

    # 方法1: 使用argmax后的二进制mask生成图像
    binary_masks = [(mask == i).squeeze().cpu().numpy() for i in range(5)]

    # 检查每个类别的二进制mask
    print("\n=== 各类别二进制mask统计 ===")
    for i in range(5):
        pixel_count = binary_masks[i].sum()
        total_pixels = binary_masks[i].size
        percentage = (pixel_count / total_pixels) * 100
        print(f"  类别 {i}: {pixel_count} 个像素 ({percentage:.2f}%)")

    to_binary_img = [
        Image.fromarray((binary_masks[i] * 255).astype(np.uint8)) for i in range(5)
    ]

    # 方法2: 如果您想查看原始置信度图
    confidence_maps = [output[0, i, :, :].cpu() for i in range(5)]
    # 对置信度图进行归一化到[0, 1]范围
    normalized_confidence = []
    for i in range(5):
        conf_map = confidence_maps[i]
        # 归一化到0-1
        conf_min = torch.min(conf_map)
        conf_max = torch.max(conf_map)
        if conf_max > conf_min:
            norm_conf = (conf_map - conf_min) / (conf_max - conf_min)
        else:
            norm_conf = conf_map - conf_min  # 如果所有值都相同
        normalized_confidence.append(norm_conf)

    to_confidence_img = [
        Image.fromarray((normalized_confidence[i].numpy() * 255).astype(np.uint8))
        for i in range(5)
    ]

    # 展示二进制mask结果 (推荐使用这个)
    for i in range(5):
        to_binary_img[i].show(title=f"Binary Mask {i}")

    # 如果您也想看置信度图
    # for i in range(5):
    #     to_confidence_img[i].show(title=f"Confidence Map {i}")