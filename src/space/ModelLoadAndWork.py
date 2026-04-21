from src.model.MultiU_NetModel import MultiU_Net
from src.model.MaskTransform import multi_class_post_process
from src.model.MaskVisualization import segmentation_visualizer
from src.dataset.FeatureExtraction import extract_features

# from MultiU_NetModel import MultiU_Net
# from MaskTransform import multi_class_post_process
# from MaskVisualization import segmentation_visualizer
# from FeatureExtraction import extract_features

import sys
import os
import torch
import numpy as np

# BEST_PERM = [1, 2, 0, 3, 4]
BEST_PERM = [3, 0, 1, 2, 4]


import torch
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def load_model():
    from src.model.MultiU_NetModel import MultiU_Net

    model = MultiU_Net(
        in_channel=22,
        depth=[3] * 5,
        depthwise_separable=False,
        combine_method='out_layer',
    ).to(device)

    # 假设模型参数文件位于 "resources/model_params" 目录下，并且命名为 "unet_branch0.pth", "unet_branch1.pth", ..., "unet_branch4.pth" 和 "out_conv.pth"
    para_dir = os.path.join(pj_root, 'resources', 'para')
    para_list = [
        os.path.join(para_dir, 'unet_branch0.pth'),
        os.path.join(para_dir, 'unet_branch1.pth'),
        os.path.join(para_dir, 'unet_branch2.pth'),
        os.path.join(para_dir, 'unet_branch3.pth'),
        os.path.join(para_dir, 'unet_branch4.pth'),
    ]
    out_para = os.path.join(para_dir, 'out_conv.pth')
    model.read_param(para_list, out_para)

    # 设置模型为评估模式（推理模式）
    model.eval()

    return model  # 一定要返回模型实例


def run_inference(model, image):
    """
    运行模型推理
    :param image: numpy格式的图像 (H, W, 3)
    :return: numpy格式的可视化结果
    """

    # 确保输入图像是正确的格式 (H, W, 3) 并且是 float32
    if image.dtype != np.float32:
        image = image.astype(np.float32)

    # 转换为PyTorch张量并调整维度 (H, W, 3) -> (3, H, W)
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()

    # 提取特征 (22, H, W)
    input_tensor = extract_features(image).to(device)

    with torch.no_grad():
        output = model(input_tensor.unsqueeze(0))  # [1, 5, H, W]

    # 获取预测mask
    pred_mask = torch.argmax(output, dim=1)  # [1, H, W]

    # 后处理
    fixed_output = multi_class_post_process(pred_mask, BEST_PERM)  # [1, H, W]

    # 准备可视化图像张量
    # 如果原始图像值域是0-255，保持不变；如果是0-1，需要转换
    if image.max() <= 1.0:
        # 如果图像是0-1范围，转换为0-255
        vis_image = (image_tensor * 255).unsqueeze(0).clamp(0, 255)  # [1, 3, H, W]
    else:
        # 如果图像已经是0-255范围
        vis_image = image_tensor.unsqueeze(0)  # [1, 3, H, W]

    # 可视化
    visualized = segmentation_visualizer(
        image=vis_image,  # [1, 3, H, W]
        mask=fixed_output.cpu(),  # [1, H, W] 去除批次维度
    )

    return visualized  # 返回可视化结果，它已经是(H, W, 3)格式