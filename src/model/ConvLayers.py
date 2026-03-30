import torch
import torch.nn as nn

# 改进版本 U-Net，使用深度可分离卷积代替传统卷积，进行二分类分割任务

class DeepwiseSeparableConv(nn.Module):
    """深度可分离卷积：Depthwise Conv -> Pointwise Conv"""

    def __init__(self, in_ch, out_ch, dropout_rate):
        super(DeepwiseSeparableConv, self).__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate, inplace=False)
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate, inplace=False)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class DoubleConv(nn.Module):
    """双卷积模块，使用深度可分离卷积替代传统卷积"""

    def __init__(self, in_ch, out_ch, dropout_rate):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            DeepwiseSeparableConv(in_ch, out_ch, dropout_rate),
            DeepwiseSeparableConv(out_ch, out_ch, dropout_rate)
        )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    """下采样模块：MaxPool -> DoubleConv"""

    def __init__(self, in_ch, out_ch, dropout_rate):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, dropout_rate)
        )

    def forward(self, x):
        return self.mpconv(x)

class Up(nn.Module):
    """上采样模块：Upsample -> Conv -> Concat -> DoubleConv"""

    def __init__(self, in_ch, out_ch, dropout_rate, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch, dropout_rate)  # in_ch = skip_connection + upsampled

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # 如果 x1 和 x2 尺寸不完全相同，进行填充使其匹配
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """输出卷积模块，使用深度可分离卷积替代传统卷积，输出通道数为 1，适用于二分类分割任务"""

    def __init__(self, in_ch, out_ch, dropout_rate):
        super(OutConv, self).__init__()
        self.conv = DeepwiseSeparableConv(in_ch, out_ch, dropout_rate)

    def forward(self, x):
        return self.conv(x)

class CBNDLayer(nn.Module):
    """卷积 -> BatchNorm -> ReLU -> Dropout"""

    def __init__(self, in_ch, out_ch, dropout_rate):
        super(CBNDLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(p=dropout_rate, inplace=False),
        )

    def forward(self, x):
        return self.conv(x)