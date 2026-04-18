from src.model.MultiU_NetModel import MultiU_Net
from src.model.MaskTransform import multi_class_post_process
from src.model.MaskVisualization import segmentation_visualizer
from src.dataset.FeatureExtraction import extract_features

# from MultiU_NetModel import MultiU_Net
# from MaskTransform import multi_class_post_process
# from MaskVisualization import segmentation_visualizer
# from FeatureExtraction import extract_features

import sys
import numpy as np

BEST_PERM = [1, 2, 0, 3, 4]

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    model = MultiU_Net(
        in_channel=22,
        depth=[3] * 5,
        depthwise_separable=False,
        combine_method='out_layer',
    ).to(device)

    import os
    para_dir = '../resources/para'
    para_list = [
        os.path.join(para_dir, 'unet_branch0.pth'),
        os.path.join(para_dir, 'unet_branch1.pth'),
        os.path.join(para_dir, 'unet_branch2.pth'),
        os.path.join(para_dir, 'unet_branch3.pth'),
        os.path.join(para_dir, 'unet_branch4.pth'),
    ]
    out_para = os.path.join(para_dir, 'out_conv.pth')
    model.read_param(para_list, out_para)

    return model

def run_inference(model, image):
    """
    先将图像提取特征
    然后用模型进行推理，得到分割掩膜
    最后产生可视化结果
    :param model: nn.Module 模型
    :param image: numpy格式的图像 (H, W, 3)
    :return: numpy格式的可视化结果
    """
    print(f"输入图像形状: {image.shape}, 类型: {type(image)}")

    # 确保输入图像是正确的格式 (H, W, 3) 并且是 float32
    if image.dtype != np.float32:
        image = image.astype(np.float32)

    # 转换为 PyTorch 张量 (H, W, 3) -> (3, H, W)
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()

    # 提取特征 (22, H, W)
    input_tensor = extract_features(image).to(device)

    print("输入图像张量形状:", image_tensor.shape, "数据类型:", image_tensor.dtype)
    print("模型输入张量形状:", input_tensor.shape, "数据类型:", input_tensor.dtype)
    sys.stdout.flush()

    with torch.no_grad():
        output = model(input_tensor.unsqueeze(0))  # [1, 5, H, W]

    print("模型输出张量形状:", output.shape, "数据类型:", output.dtype)
    sys.stdout.flush()

    # 对输出进行后处理
    # output shape: [1, 5, H, W] -> [1, H, W] -> [H, W] -> 应用BEST_PERM
    pred_mask = torch.argmax(output, dim=1)  # [1, H, W]

    # 直接对 [1, H, W] 形状的预测结果应用后处理，不需要额外的维度操作
    fixed_output = multi_class_post_process(pred_mask, BEST_PERM)  # [1, H, W]

    # 准备可视化所需的图像张量 (1, 3, H, W)
    vis_image = image_tensor.unsqueeze(0)  # [1, 3, H, W]

    # 可视化 - fixed_output 是 [1, H, W]，需要与图像维度匹配
    visualized = segmentation_visualizer(
        image=vis_image,
        mask=fixed_output.cpu(),  # [1, H, W]
        # mask=pred_mask.cpu(),  # [1, H, W]
    )

    # 检查visualized的形状并正确处理
    result = visualized
    if result.ndim == 4 and result.shape[0] == 1:  # [1, C, H, W] -> [C, H, W]
        result = result.squeeze(0)
    elif result.ndim == 3 and result.shape[0] == 3:  # [3, H, W] -> [H, W, 3]
        result = np.transpose(result, (1, 2, 0))

    print("最终可视化结果形状:", result.shape)
    return result