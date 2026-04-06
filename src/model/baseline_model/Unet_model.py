import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    """
    经典 U-Net 架构的 PyTorch 实现

    该模型接受 3 通道输入 (C_in=3)，经过下采样和上采样后，
    输出 n 个通道 (num_classes=n) 的特征图。最后通过 Softmax
    计算每个像素属于各个类别的概率，返回概率最高的类别索引，
    最终输出一个形状为 [batch_size, 1, H, W] 的分割图。
    """

    def __init__(self, num_classes=2, in_channels=3, dropout_rate=0.1, bilinear=True):
        """
        初始化 U-Net 模型。

        Args:
            num_classes (int): 分割的类别总数 (输出通道数)。例如，二分类为2。
            in_channels (int): 输入图像的通道数。彩色图默认为3。
            dropout_rate (float): Dropout 层的概率。默认为 0.1。
            bilinear (bool): 上采样方式。True 使用双线性插值 (更轻量)，False 使用转置卷积。
        """
        super(UNet, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.dropout_rate = dropout_rate
        self.bilinear = bilinear

        # 编码器 (下采样路径)
        self.inc = DoubleConv(in_channels, 64, dropout_rate)
        self.down1 = Down(64, 128, dropout_rate)
        self.down2 = Down(128, 256, dropout_rate)
        self.down3 = Down(256, 512, dropout_rate)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, dropout_rate)

        # 解码器 (上采样路径)
        self.up1 = Up(1024, 512 // factor, dropout_rate, bilinear)
        self.up2 = Up(512, 256 // factor, dropout_rate, bilinear)
        self.up3 = Up(256, 128 // factor, dropout_rate, bilinear)
        self.up4 = Up(128, 64, dropout_rate, bilinear)

        # 输出层
        self.outc = OutConv(64, num_classes)

    def forward(self, x):
        """
        前向传播过程。

        Args:
            x (Tensor): 形状为 [batch_size, in_channels, H, W] 的输入张量。

        Returns:
            Tensor: 形状为 [batch_size, 1, H, W] 的分割结果，每个像素值是其预测的类别索引。
        """
        # 编码器路径
        x1 = self.inc(x)  # [B, 64, H, W]
        x2 = self.down1(x1)  # [B, 128, H/2, W/2]
        x3 = self.down2(x2)  # [B, 256, H/4, W/4]
        x4 = self.down3(x3)  # [B, 512, H/8, W/8]
        x5 = self.down4(x4)  # [B, 1024//factor, H/16, W/16]

        # 解码器路径
        x = self.up1(x5, x4)  # [B, 512//factor, H/8, W/8]
        x = self.up2(x, x3)  # [B, 256//factor, H/4, W/4]
        x = self.up3(x, x2)  # [B, 128//factor, H/2, W/2]
        x = self.up4(x, x1)  # [B, 64, H, W]

        # 输出层得到各类别概率 logit
        logits = self.outc(x)  # [B, num_classes, H, W]

        # 计算 Softmax 并取最大值的索引，得到最终分割图
        # dim=1 是对类别维度 (num_classes) 进行操作
        seg_map = torch.argmax(F.softmax(logits, dim=1), dim=1, keepdim=True)
        # seg_map.shape -> [B, 1, H, W], dtype=torch.int64

        return logits
        # return seg_map


# --- U-Net 的基础组件 ---

class DoubleConv(nn.Module):
    """两次 [Conv -> BatchNorm -> ReLU -> Dropout] 操作."""

    def __init__(self, in_ch, out_ch, dropout_rate):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate, inplace=False),

            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate, inplace=False)
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
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """输出层：1x1 卷积，将特征图通道数映射到类别数."""

    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x)

def print_model_params(model):
    print("Model parameter summary:")
    print("-" * 60)
    total = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            numel = param.numel()
            total += numel
            print(f"{name:<50} {numel:>12,}")
    print("-" * 60)
    print(f"{'Total trainable parameters':<50} {total:>12,}")

# --- 示例用法 ---
if __name__ == "__main__":
    # 创建一个 4 类分割的 U-Net 模型
    model = UNet(num_classes=5, dropout_rate=0.1, bilinear=True)

    # 创建一个模拟的输入批次 (batch_size=1, channels=3, height=256, width=256)
    dummy_input = torch.randn(1, 3, 256, 256)

    # 模型推理
    with torch.no_grad():  # 推理时不需要梯度
        output_segmentation = model(dummy_input)

    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output_segmentation.shape}")  # 应为 [1, 1, 256, 256]
    print(f"输出数据类型: {output_segmentation.dtype}")  # 应为 torch.int64
    print(f"输出最大值: {output_segmentation.max().item()}, 最小值: {output_segmentation.min().item()}")  # 应为 3 和 0
    print("模型构建和推理成功！")

    print_model_params(model)
