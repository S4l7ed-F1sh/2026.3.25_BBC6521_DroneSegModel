import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from src.model.MultiU_NetModel import MultiBranchU_Net
from src.dataset.DroneSegDataSet import MyDataset


class OutConvDs(Dataset):
    def __init__(self, image_dir, label_dir, model_param_list, transform=None, ds_not_in_resources=False,
                 data_enforcement=False):
        super().__init__()

        # --- 配置缓存路径 ---
        # 假设当前文件是在项目根目录下的某个子文件夹运行，或者你需要根据实际情况调整这个相对路径
        # 如果不确定根目录，可以使用绝对路径，例如: "/home/user/project/resources/dataset/temp"
        self.cache_dir = "resources/dataset/temp"

        # 确保目录存在
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            print(f"创建缓存目录: {self.cache_dir}")

        # 1. 初始化原始数据集
        self.raw_dataset = MyDataset(
            image_dir=image_dir,
            label_dir=label_dir,
            transform=transform,
            ds_not_in_resources=ds_not_in_resources,
            data_enforcement=data_enforcement,
        )

        # 2. 初始化模型 (仅用于预处理阶段)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MultiBranchU_Net(
            in_channel=22,
            depth=[3] * 5,
            bilinear=True,
            n_classes=5,
            depthwise_separable=False,
        )
        self.model.read_param(model_param_list)
        self.model.to(self.device)
        self.model.eval()

        # --- 核心逻辑：检查缓存是否已存在 ---
        # 我们约定：如果目录下有文件，说明已经预处理过了，直接跳过推理
        existing_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pt')]

        if len(existing_files) == len(self.raw_dataset):
            print(f"✅ 发现完整缓存 ({len(existing_files)} 个文件)，跳过 Encoder 推理，直接加载索引。")
        else:
            # 缓存不完整或为空，需要重新运行推理
            if len(existing_files) > 0:
                print(
                    f"⚠️ 警告：缓存目录中有 {len(existing_files)} 个文件，但数据集有 {len(self.raw_dataset)} 个。将重新生成所有缓存。")
                # 可选：清空旧缓存以防混淆
                for f in os.listdir(self.cache_dir): os.remove(os.path.join(self.cache_dir, f))

            print(f"🚀 开始预处理数据 (Encoder推理中)... 目标目录: {self.cache_dir}")

            with torch.no_grad():
                for idx in tqdm(range(len(self.raw_dataset))):
                    # 1. 获取原始数据
                    feature_image, label, raw_image = self.raw_dataset.__getitem__(idx)

                    # 2. 推理
                    feature_image = feature_image.to(self.device)
                    output = self.model(feature_image.unsqueeze(0))
                    model_output = output.squeeze(0).detach().cpu()  # 转回 CPU

                    # 3. 保存到磁盘
                    # 格式: resources/dataset/temp/{idx}.pt
                    cache_path = os.path.join(self.cache_dir, f"{idx}.pt")

                    # 保存为一个字典，包含 encoder输出、标签和原图
                    # 注意：如果 raw_image 很大，你也可以选择只保存 model_output 和 label
                    torch.save({
                        'output': model_output,
                        'label': label,
                        'raw_image': raw_image
                    }, cache_path)

            print("💾 数据预处理并保存完成！")

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        # 从磁盘读取对应 idx 的文件
        cache_path = os.path.join(self.cache_dir, f"{idx}.pt")

        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"缓存文件不存在: {cache_path}. 请检查 __init__ 是否运行成功。")

        # 加载数据
        data = torch.load(cache_path)

        return data['output'], data['label'], data['raw_image']