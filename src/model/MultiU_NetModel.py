import os.path

import torch
import torch.nn as nn

from src.model.U_NetModel import U_Net
from src.model.ConvLayers import CBNDLayer

class MultiU_Net(nn.Module):
    """多分支 U-Net 模型，将 n 分类任务分割为 n 个二分类任务，每个分支都有一个 U-Net 模型，输出 1 通道信息，"""

    def __init__(
            self,
            in_channel,
            depth:list,
            dropout_rate=0.1,
            bilinear=True,
            n_classes=5,
            depthwise_separable=True,
    ):
        super(MultiU_Net, self).__init__()

        self.multi_branch = MultiBranchU_Net(
            in_channel=in_channel,
            dropout_rate=dropout_rate,
            bilinear=bilinear,
            depth=depth,
            n_classes=n_classes,
            depthwise_separable=depthwise_separable,
        )
        self.out_layer = OutLayer(in_ch=n_classes*2, out_ch=n_classes, dropout_rate=dropout_rate)

    def forward(self, x):
        multi_branch_output = self.multi_branch(x)  # [B, n_classes * 2, H, W]
        out = self.out_layer(multi_branch_output)  # [B, n_classes, H, W]
        return out  # [B, n_classes, H, W]

    def read_param(self, branchs: list, out_layer: str):
        """读取模型参数，branchs 是一个包含 n_classes 个文件路径的列表，每个文件路径对应一个分支 U-Net 模型的参数文件，out_layer 是输出层参数文件的路径"""
        self.multi_branch.read_param(branchs)
        self.out_layer.read_param(out_layer)

class MultiBranchU_Net(nn.Module):
    """多分支 U-Net 模型，每个分支都有一个 U-Net 模型，输出 n_classes 通道信息，进行 n 分类任务"""

    def __init__(
            self,
            in_channel,
            depth:list,
            dropout_rate=0.1,
            bilinear=True,
            n_classes=5,
            depthwise_separable=True,
    ):
        super(MultiBranchU_Net, self).__init__()
        self.in_channel = in_channel
        self.dropout_rate = dropout_rate
        self.bilinear = bilinear
        self.depth = depth
        self.n_classes = n_classes

        self.unet_branches = nn.ModuleList([
            U_Net(
                in_channel=in_channel,
                dropout_rate=dropout_rate,
                bilinear=bilinear,
                depth=depth[i],
                depthwise_separable=depthwise_separable,
            )
            for i in range(n_classes)
        ])

    def forward(self, x):
        branch_outputs = []
        for i in range(self.n_classes):
            branch_output = self.unet_branches[i](x)  # [B, 2, H, W]
            branch_outputs.append(branch_output)
        combined = torch.cat(branch_outputs, dim=1)  # [B, n_classes * 2, H, W]

        return combined  # [B, n_classes * 2, H, W]

    def read_param(self, files_paths: list):
        """读取多个分支 U-Net 模型的参数，files_paths 是一个包含 n_classes 个文件路径的列表，每个文件路径对应一个分支 U-Net 模型的参数文件"""
        for i in range(self.n_classes):
            self.unet_branches[i].load_state_dict(torch.load(files_paths[i]))

class OutLayer(nn.Module):
    """输出层，将 n_classes * 2 通道信息转换为 n_classes 通道信息"""

    def __init__(self, n_classes=5, dropout_rate=0.1):
        super(OutLayer, self).__init__()

        in_ch = n_classes * 2
        out_ch = n_classes

        self.conv1 = CBNDLayer(in_ch=in_ch, out_ch=64, dropout_rate=dropout_rate)
        self.conv2 = CBNDLayer(in_ch=64, out_ch=out_ch, dropout_rate=dropout_rate)

    def forward(self, x):
        return self.conv2(self.conv1(x))

    def read_param(self, file_path: str):
        """读取输出层的参数，file_path 是输出层参数文件的路径"""
        self.load_state_dict(torch.load(file_path))


if __name__ == "__main__":
    net = MultiBranchU_Net(in_channel=22, depth=[3] * 5, depthwise_separable=False)

    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 返回项目根目录（向上两级：src/model -> src -> 项目根目录）
    project_root = os.path.dirname(os.path.dirname(current_dir))
    # 构建para目录路径
    para_dir = os.path.join(project_root, 'resources', 'para')

    para_list = [
        os.path.join(para_dir, 'class0_phase5.pth'),
        os.path.join(para_dir, 'class1_phase5.pth'),
        os.path.join(para_dir, 'class2_phase5.pth'),
        os.path.join(para_dir, 'class3_phase5.pth'),
        os.path.join(para_dir, 'class4_phase5.pth'),
    ]

    print(os.path.exists(para_list[0]))
    net.read_param(para_list)
