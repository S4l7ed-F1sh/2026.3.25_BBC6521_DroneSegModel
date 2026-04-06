import os

from pj_root import PROJECT_ROOT
from src.dataloaders.dataset00 import MyDataset
from src.utils.ds_check import check_ds
from torch.utils.data import random_split, DataLoader
from Unet_model import UNet
import torch

CURRENT_DIR = os.getcwd()
RES_DIR = os.path.join(PROJECT_ROOT, 'resources')

def main():
    model = UNet(num_classes=5)

    dataset_path = check_ds(
        ds_name="dataset00",
        kaggle_path=""
    )
    print("数据集路径:", dataset_path)

    dataset = MyDataset("dataset/dataset00/images", "dataset/dataset00/labels")

    # train_set, test_set = random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
    # train_loader = DataLoader(dataset=train_set, batch_size=4, shuffle=True)
    # test_loader = DataLoader(dataset=test_set, batch_size=4, shuffle=False)

    print(f"数据集总样本数: {len(dataset)}")
    # print(f"训练集样本数: {len(train_set)}")
    # print(f"测试集样本数: {len(test_set)}")

    img1, lbl1, _ = dataset.__getitem__(0)

    print("图像1的形状:", img1.shape)
    print("标签1的形状:", lbl1.shape)

    img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float()  # 转换为张量并添加批次维度
    ans1 = model(img1_tensor)  # 添加批次维度
    print("模型输出的形状:", ans1.shape)

if __name__ == '__main__':
    main()
