import torch.nn as nn
import torch
from src.model.Model1.ConvAndReluLayer import CRLayer, CBNRLayer

# 模型的空洞卷积残差模块
# 实现思路：输入通道数为 C 的图片进行空洞率为 1, 2, 4 的卷积操作，得到三个尺寸相同的特征图
# 然后将这三个特征图进行相加，得到一个通道数为 C*3 的特征图，然后通过一个 1*1 的卷积将通道数调整回 C
# 最后，将输入的原图像与上一步得到的特征图进行相加，得到最终的输出特征图
class AtrousResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AtrousResidualBlock, self).__init__()

        self.d_conv1 = CBNRLayer(in_channels=out_channels, out_channels=out_channels, kernel_size=3, dilation=1, padding=1)
        self.d_conv2 = CBNRLayer(in_channels=out_channels, out_channels=out_channels, kernel_size=3, dilation=2, padding=2)
        self.d_conv3 = CBNRLayer(in_channels=out_channels, out_channels=out_channels, kernel_size=3, dilation=3, padding=3)

        self.adj_conv0 = CRLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0)
        self.adj_conv1 = CRLayer(in_channels=out_channels*3, out_channels=out_channels, kernel_size=1, padding=0)

    def forward(self, input):
        input = self.adj_conv0(input)

        # 通过三个不同空洞率的卷积操作得到三个特征图
        d1 = self.d_conv1(input)
        d2 = self.d_conv2(input)
        d3 = self.d_conv3(input)

        # 将三个特征图进行相加
        combined = torch.cat((d1, d2, d3), dim=1)  # 在通道维度上进行拼接

        # 调整通道数
        adjusted = self.adj_conv1(combined)

        # 将输入的原图像与调整后的特征图进行相加
        output = input + adjusted

        del d1, d2, d3, combined, adjusted  # 释放内存

        return output


if __name__ == "__main__":
    # 测试代码
    block = AtrousResidualBlock(in_channels=96, out_channels=32)
    input_tensor1 = torch.randn(1, 32, 256, 256)  # 模拟输入特征图
    input_tensor2 = torch.randn(1, 96, 256, 256)  # 模拟输入特征图

    output_tensor1 = block(input_tensor1)
    output_tensor2 = block(input_tensor2)
    print(output_tensor1.shape)  # 应该输出 torch.Size([1, 32, 256, 256])
    print(output_tensor2.shape)  # 应该输出 torch.Size([1, 32, 256, 256])
