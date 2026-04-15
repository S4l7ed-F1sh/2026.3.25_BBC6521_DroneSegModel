import os
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from src.dataset.FeatureExtraction import extract_features
from torchvision import transforms
from src.dataset.DataReinforcement import data_augmentation
import torch

current_file_dir = os.path.dirname(os.path.abspath(__file__))
print(f"当前脚本路径: {current_file_dir}")
project_root = os.path.abspath(os.path.join(current_file_dir, '..', '..'))
print(f"项目根路径: {project_root}")

class MyDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, ds_not_in_resources=False, data_enforcement=False):
        if not ds_not_in_resources:
            image_dir = os.path.join(project_root, 'resources/dataset', image_dir)
            label_dir = os.path.join(project_root, 'resources/dataset', label_dir)

        if not os.path.exists(image_dir) or not os.path.exists(label_dir):
            print(f"警告：检测到数据集图像目录或标签目录不存在！请检查以下路径是否正确：")
            print(f"图像目录路径: {image_dir}")
            print(f"标签目录路径: {label_dir}")
            raise FileNotFoundError("数据集图像目录或标签目录不存在，请检查路径是否正确！")

        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.data_enforcement = data_enforcement

        image_files_list = os.listdir(self.image_dir)
        label_files_list = os.listdir(self.label_dir)

        image_set = set(image_files_list)
        label_set = set(label_files_list)

        if image_set != label_set:
            print("警告：图像文件和标签文件不匹配！请检查目录中的文件是否正确对应。")
            matched_set = image_set & label_set
            self.files = list(matched_set)  # 转为列表以便后续索引操作

            unmatched_images = image_set - matched_set
            unmatched_labels = label_set - matched_set

            print(
                f"在 image 文件夹中，发现的不匹配文件: {', '.join(list(unmatched_images)[:3]) + (f' 等 {len(unmatched_images)} 个文件' if len(unmatched_images) > 3 else '')}")
            print(
                f"在 label 文件夹中，发现的不匹配文件: {', '.join(list(unmatched_labels)[:3]) + (f' 等 {len(unmatched_labels)} 个文件' if len(unmatched_labels) > 3 else '')}")
        else:
            self.files = list(image_set)

        print(f"成功匹配文件: {', '.join(self.files[:3]) + (f' 等 {len(self.files)} 个文件' if len(self.files) > 3 else '')}")

    def __len__(self):
        if not self.data_enforcement:
            return len(self.files)
        else:
            return len(self.files) * 4  # 强制扩充数据集大小为原来的4倍，以增加训练样本数量

    def __getitem__(self, idx):
        idx %= len(self.files)

        image_path = os.path.join(self.image_dir, self.files[idx])
        label_path = os.path.join(self.label_dir, self.files[idx])

        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path)

        # 确保标签是 'P' 模式或 'L' 模式
        if label.mode != 'P' and label.mode != 'L':
            label = label.convert('P')

        # 数据增强 (注意：如果你的 transform 包含 ToTensor，这里会出问题)
        # 假设 data_augmentation 是基于 numpy 的 (如 albumentations 或自定义 numpy 操作)
        if self.data_enforcement:
            img_np, lbl_np = data_augmentation(np.array(image), np.array(label))
        else:
            img_np, lbl_np = np.array(image), np.array(label)

        # --- 修改部分开始 ---

        # 1. 处理图像：使用 transform (包含 ToTensor 和 Normalize)
        if self.transform:
            img = self.transform(img_np)  # 此时 img 是 [0,1] 或归一化后的 float
        else:
            img = torch.from_numpy(img_np).permute(2, 0, 1).float()

        # 2. 处理标签：手动转换，绝对不要除以 255
        # 将 numpy 数组转为 torch tensor
        lbl = torch.from_numpy(lbl_np)
        # 确保类型是 Long，因为交叉熵损失函数需要 Long 类型
        lbl = lbl.long()

        # --- 修改部分结束 ---

        ext_feature = extract_features(img_np)  # 注意：这里传 numpy 还是 tensor 取决于你的 extract_features 实现

        return ext_feature, lbl, img

    def check_sample(self):
        rd_idx = np.random.randint(len(self.files))

        fet, lbl, img = self.__getitem__(rd_idx)

        print("检查样本的特征图像尺寸: ", fet.shape)
        print("检查样本的原始图像尺寸: ", img.shape)
        print("检查样本的标签图像尺寸: ", lbl.shape)

        fet_w, fet_h = fet.shape[2], fet.shape[1]
        lbl_w, lbl_h = lbl.shape[1], lbl.shape[0]
        img_w, img_h = img.shape[2], img.shape[1]

        fet_c = fet.shape[0] if len(img.shape) == 3 else 1
        img_c = img.shape[0] if len(img.shape) == 3 else 1
        lbl_c = lbl.shape[0] if len(lbl.shape) == 3 else 1

        lbl_u = np.unique(lbl)
        # img_u = np.unique(img)

        print('=' * 70)
        print(f"随机检查样本:")
        print(f"feature 图像尺寸: {fet_w}x{fet_h}，通道数: {fet_c}")
        print(f"label 图像尺寸: {lbl_w}x{lbl_h}，通道数: {lbl_c}")
        print(f"label 图像中值域: {lbl_u}")
        print(f"image 图像尺寸: {img_w}x{img_h}，通道数: {img_c}")
        # print(f"image 图像中值域: {img_u}")
        print('=' * 70)

    def get_file_name(self, idx):
        idx %= len(self.files)
        return self.files[idx]

def convert_to_binary_label(label_tensor, class_idx):
    """
    将多分类 Label (B, H, W) 转换为特定类别的二分类 Label (B, H, W)。

    参数:
        label_tensor (torch.Tensor): 原始标签，形状为 (B, H, W)，值域为 0~n-1，类型需为 Long
        class_idx (int): 想要提取的目标类别索引 (0 ~ n-1)

    返回:
        torch.Tensor: 二分类标签，形状为 (B, H, W)，值为 0 (不是该类) 或 1 (是该类)
    """
    # 1. 比较操作：生成布尔掩码
    # 如果像素值等于 class_idx，则为 True，否则为 False
    # 结果形状：(B, H, W)
    binary_mask = (label_tensor == class_idx)

    # 2. 类型转换：将布尔值转换为整数 (0 和 1)
    binary_mask = binary_mask.long()

    return binary_mask

def main():
    image_dir = 'drone_seg_dataset/classes_dataset/classes_dataset/original_images'
    label_dir = 'drone_seg_dataset/classes_dataset/classes_dataset/label_images_semantic'

    dataset = MyDataset(
        image_dir,
        label_dir,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ])
    )

    print(f"数据集大小: {len(dataset)}")

    dataset.check_sample()
    dataset.check_sample()
    dataset.check_sample()

if __name__ == '__main__':
    main()
