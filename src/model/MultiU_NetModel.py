import torch
import torch.nn as nn

from src.model.U_NetModel import U_Net
from src.model.ConvLayers import CBNDLayer

class MultiU_Net(nn.Module):
    """多分支 U-Net 模型，将 n 分类任务分割为 n 个二分类任务，每个分支都有一个 U-Net 模型，输出 1 通道信息，"""

    def __init__(
            self,
            in_channel,
            dropout_rate=0.1,
            bilinear=True,
            depth=list,
            n_classes=5,
    ):
        super(MultiU_Net, self).__init__()
        self.in_channel = in_channel
        self.dropout_rate = dropout_rate
        self.bilinear = bilinear
        self.depth = depth
        self.n_classes = n_classes

        self.unet_branches = nn.ModuleList([
            U_Net(in_channel=in_channel, dropout_rate=dropout_rate, bilinear=bilinear, depth=depth[i])
            for i in range(n_classes)
        ])

        self.combine_conv0 = CBNDLayer(in_ch=n_classes, out_ch=64, dropout_rate=dropout_rate)
        self.combine_conv1 = CBNDLayer(in_ch=64, out_ch=n_classes, dropout_rate=dropout_rate)

    def forward(self, x):
        branch_outputs = []
        for i in range(self.n_classes):
            branch_output = self.unet_branches[i](x)  # [B, 1, H, W]
            branch_outputs.append(branch_output)
        combined = torch.cat(branch_outputs, dim=1)  # [B, n_classes, H, W]

        combined = self.combine_conv0(combined)  # [B, 64, H, W]
        combined = self.combine_conv1(combined)  # [B, n_classes, H, W]

        return combined  # [B, n_classes, H, W]
