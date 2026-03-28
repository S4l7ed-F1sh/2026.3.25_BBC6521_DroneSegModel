import torch
import torch.nn as nn
from absl.logging import level_info

from src.model.MultiResolutionLevel import MultiResolutionLevel
from src.model.ConvLayers import CRLayer, CBNRLayer, CBNDLayer


# 输入的图片有 32 个通道，输出的图片有 n_classes 个通道
# 模型的结构设计思路：分为 layers 层 MultiResolutionLevel，每层的最大通道数为 max_channels。
# 第 i 层的输入通道为 input_channels // (2**i)，输出通道为 max_channels。分辨率为输入图片分辨率的 (2**i) 分之一。
class DroneSegModel(torch.nn.Module):
    def __init__(self, n_classes=5, layers=4, level_channels=32, feat_channels=22):
        super(DroneSegModel, self).__init__()

        self.n_classes = n_classes
        self.layers = layers
        self.level_channels = level_channels
        self.feat_channels = feat_channels

        self.c_adjust0 = CRLayer(in_channels=feat_channels, out_channels=level_channels, kernel_size=1, padding=0)

        self.levels = nn.ModuleList(
            [
                MultiResolutionLevel()
                for _ in range(layers)
            ]
        )

        self.combiner = nn.ModuleList(
            [
                CBNDLayer(in_channels=level_channels*2, out_channels=level_channels, kernel_size=3, padding=1)
                for _ in range(layers)
            ]
        )

        self.upsampler = nn.ModuleList(
            [
                CBNDLayer(in_channels=level_channels*2, out_channels=level_channels, kernel_size=3, padding=1)
                for _ in range(layers-1)
            ]
        )

        self.final_conv = nn.Conv2d(in_channels=level_channels, out_channels=n_classes, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.c_adjust0(x)

        layer_inputs = [x]
        for i in range(self.layers-1):
            layer_inputs.append(nn.AvgPool2d(kernel_size=2, stride=2)(layer_inputs[-1]))

        layer_outputs = []
        for i in range(self.layers):
            layer_outputs.append(
                self.levels[i](layer_inputs[i])
            )

        combined_outputs = []
        for i in range(self.layers):
            combined = torch.cat([layer_outputs[i], layer_inputs[i]], dim=1)
            combined = self.combiner[i](combined)
            combined_outputs.append(combined)

        to_combine = combined_outputs[-1]
        for i in range(self.layers-1):
            print(f"Combining layer {self.layers-1-i} with layer {self.layers-2-i}")
            print(f"to_combine shape: {to_combine.shape}, combined_outputs[{self.layers-2-i}] shape: {combined_outputs[self.layers-2-i].shape}")

            upsampled = nn.functional.interpolate(to_combine, scale_factor=2, mode='bilinear', align_corners=False)

            print(f"Upsampled shape: {upsampled.shape}")

            to_combine = self.upsampler[i](
                torch.cat([upsampled, combined_outputs[self.layers - i - 2]], dim=1)
            )

        return self.final_conv(to_combine)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model = DroneSegModel(n_classes=5, layers=4)
    model = model.to(device)  # 👈 移动模型到 GPU
    model.train()

    criterion = nn.CrossEntropyLoss()

    # 创建张量并移动到 GPU
    x = torch.randn(1, 22, 736, 960, device=device)
    y = torch.randint(0, 5, (1, 736, 960), device=device)

    output = model(x)
    print(output.shape)
    loss = criterion(output, y)
    loss.backward()

    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        print(f"Peak GPU memory: {peak_mem:.2f} MB")
    else:
        print("Running on CPU, no GPU memory used.")

if __name__ == "__main__":
    main()
