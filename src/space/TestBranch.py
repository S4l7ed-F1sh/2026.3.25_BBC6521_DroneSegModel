from src.model.MultiU_NetModel import MultiBranchU_Net
from src.dataset.FeatureExtraction import extract_features

import torch
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

model = MultiBranchU_Net(
    in_channel=22,
    depth=[3] * 5,
    depthwise_separable=False,
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
model.read_param(para_list)

model.eval()

img_path = './150.png'
from PIL import Image
import numpy as np

img = Image.open(img_path)
img_np = np.array(img)

input_tensor = extract_features(img_np)

with torch.no_grad():
    input_tensor = input_tensor.unsqueeze(0).to(device)  # 添加 batch 维度并移动到设备
    output = model(input_tensor)  # [1, n_classes * 2, H, W]
    print("输出张量形状:", output.shape)

class_mask = [
    output[:, 0:2, :, :],   # 类别 0 的分支输出
    output[:, 2:4, :, :],   # 类别 1 的分支输出
    output[:, 4:6, :, :],   # 类别 2 的分支输出
    output[:, 6:8, :, :],   # 类别 3 的分支输出
    output[:, 8:10, :, :]   # 类别 4 的分支输出
]

# 对每个类别执行 argmax，然后去除通道维度和 batch 维度，最后保存为黑白图。
for i in range(5):
    class_output = class_mask[i]  # [1, 2, H, W]
    class_mask_i = torch.argmax(class_output, dim=1).squeeze(0).cpu().numpy()  # [H, W]
    class_mask_i_img = Image.fromarray((class_mask_i * 255).astype(np.uint8))  # 转换为黑白图
    class_mask_i_img.save(f'class_{i}_mask.png')  # 保存图像
    print(f"类别 {i} 的 mask 已保存为 class_{i}_mask.png")
