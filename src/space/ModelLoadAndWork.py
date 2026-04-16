# from src.model.MultiU_NetModel import MultiU_Net
# from src.model.MaskTransform import multi_class_post_process
# from src.model.MaskVisualization import segmentation_visualizer
# from src.dataset.FeatureExtraction import extract_features

from MultiU_NetModel import MultiU_Net
from MaskTransform import multi_class_post_process
from MaskVisualization import segmentation_visualizer
from FeatureExtraction import extract_features

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
    para_dir = './'
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
    :param image: numpy格式的图像
    :return: numpy格式的可视化结果
    """
    image_tensor = torch.from_numpy(image).float()  # 转为 float 类型的 tensor
    input_tensor = extract_features(image).to(device) # [1, 22, H, W]

    with torch.no_grad():
        output = model(input_tensor.unsqueeze(0))  # [1, 1, H, W]

    fixed_output = multi_class_post_process(torch.argmax(output, dim=1), BEST_PERM)

    # 可视化
    visualized = segmentation_visualizer(
        image=image_tensor.unsqueeze(0),
        mask=fixed_output.cpu(),  # 去除批次维度
    )

    return visualized.squeeze(0).cpu().numpy()  # 转回 numpy 格式，去除批次维度
