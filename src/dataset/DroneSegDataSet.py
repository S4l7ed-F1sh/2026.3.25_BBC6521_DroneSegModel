import os
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from src.dataset.FeatureExtraction import extract_features
from torchvision import transforms
import torch

current_file_dir = os.path.dirname(os.path.abspath(__file__))
print(f"当前脚本路径: {current_file_dir}")
project_root = os.path.abspath(os.path.join(current_file_dir, '..', '..'))
print(f"项目根路径: {project_root}")

class MyDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, ds_not_in_resources=False):
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
        return len(self.files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.files[idx])
        label_path = os.path.join(self.label_dir, self.files[idx])

        image = Image.open(image_path).convert('RGB')

        label = Image.open(label_path)
        if label.mode != 'P':
            print(f"警告：标签图像 {self.files[idx]} 的模式不是 'P'，请检查标签图像是否正确。")
            label = label.convert('P')

        img = np.array(image)
        lbl = np.array(label, dtype=np.int64)

        ext_feature = extract_features(img)
        # print("提取的特征维度: ", ext_feature.shape)

        if self.transform:
            img = self.transform(img)
            lbl = self.transform(lbl).squeeze(0)  # 去掉单通道维度，保持与原始标签图像一致的维度

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


import torch


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
