import torch
import torch.nn as nn
from src.model.ConvLayers import DoubleConv, Down, Up, OutConv

class U_Net(nn.Module):
    """U-Net 模型，使用深度可分离卷积代替传统卷积，输出 2 通道信息，进行二分类任务，可以改变模型下采样的深度，深度范围为 2-4"""

    def __init__(self, in_channel, out_channel=2, dropout_rate=0.1, bilinear=True, depth=4, depthwise_separable=True):
        """
            初始化 U-Net 模型。

            :param in_channel: 输入图像的通道数，默认为 3 (RGB 图像)
            :param dropout_rate: Dropout 层的概率，默认为 0.1
            :param bilinear: 上采样方式，True 使用双线性插值，False 使用转置卷积，默认为 True
            :param depth: 模型下采样的深度，范围为 2-4，默认为 4
        """
        super(U_Net, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.dropout_rate = dropout_rate
        self.bilinear = bilinear
        self.depth = depth

        if self.depth < 2 or self.depth > 4:
            raise ValueError("Depth must be between 2 and 4.")

        factor = 2 if bilinear else 1

        # 编码器 (下采样路径)
        self.inc = DoubleConv(in_channel, 64, dropout_rate, depthwise_separable=depthwise_separable)

        self.down1 = Down(in_ch=64, out_ch=128, dropout_rate=dropout_rate, depthwise_separable=depthwise_separable)

        if depth == 2:
            self.down2 = Down(in_ch=128, out_ch=256 // factor, dropout_rate=dropout_rate, depthwise_separable=depthwise_separable)
        else:
            self.down2 = Down(in_ch=128, out_ch=256, dropout_rate=dropout_rate, depthwise_separable=depthwise_separable)
        if depth == 3:
            self.down3 = Down(in_ch=256, out_ch=512 // factor, dropout_rate=dropout_rate, depthwise_separable=depthwise_separable)
        elif depth > 3:
            self.down3 = Down(in_ch=256, out_ch=512, dropout_rate=dropout_rate, depthwise_separable=depthwise_separable)
        if depth == 4:
            self.down4 = Down(in_ch=512, out_ch=1024 // factor, dropout_rate=dropout_rate, depthwise_separable=depthwise_separable)

        # 解码器 (上采样路径)
        if depth == 4:
            self.up4 = Up(in_ch=1024, out_ch=512 // factor, dropout_rate=dropout_rate, depthwise_separable=depthwise_separable)
        if depth >= 3:
            self.up3 = Up(in_ch=512, out_ch=256 // factor, dropout_rate=dropout_rate, depthwise_separable=depthwise_separable)
        self.up2 = Up(in_ch=256, out_ch=128 // factor, dropout_rate=dropout_rate, depthwise_separable=depthwise_separable)
        self.up1 = Up(in_ch=128, out_ch=64, dropout_rate=dropout_rate, depthwise_separable=depthwise_separable)

        # 输出 2 通道，进行二分类任务
        self.outc = OutConv(in_ch=64, out_ch=out_channel, dropout_rate=dropout_rate, depthwise_separable=depthwise_separable)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)

        if self.depth == 2:
            x3 = self.down2(x2)
            x = self.up2(x3, x2)
        else:
            x3 = self.down2(x2)

        if self.depth == 3:
            x4 = self.down3(x3)
            x = self.up3(x4, x3)
        elif self.depth > 3:
            x4 = self.down3(x3)

        if self.depth == 4:
            x5 = self.down4(x4)
            x = self.up4(x5, x4)

        if self.depth > 3:
            x = self.up3(x, x3)

        if self.depth > 2:
            x = self.up2(x, x2)

        x = self.up1(x, x1)
        logits = self.outc(x)
        return logits
