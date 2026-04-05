from torch.utils.data import Dataset
from src.model.MultiU_NetModel import MultiBranchU_Net
from src.dataset.DroneSegDataSet import MyDataset

import torch

class OutConvDs(Dataset):
    def __init__(self, image_dir, label_dir, model_param_list, transform=None, ds_not_in_resources=False, data_enforcement=False):
        super().__init__()
        self.dataset = MyDataset(
            image_dir=image_dir,
            label_dir=label_dir,
            transform=transform,
            ds_not_in_resources=ds_not_in_resources,
            data_enforcement=data_enforcement,
        )

        self.model = MultiBranchU_Net(
            in_channel=22,
            depth=[3] * 5,
            bilinear=True,
            n_classes=5,
            depthwise_separable=False,
        )
        self.model.read_param(model_param_list)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        feature_image, label, raw_image = self.dataset.__getitem__(idx)

        feature_image = feature_image.to(self.device)
        with torch.no_grad():
            output = self.model(feature_image.unsqueeze(0))  # 添加 batch 维度
        model_output = output.squeeze(0)  # 去掉 batch 维度
        model_output = model_output.detach().cpu()

        return model_output, label, raw_image
