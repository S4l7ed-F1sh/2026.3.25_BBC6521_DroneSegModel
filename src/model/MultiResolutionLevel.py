import torch.nn as nn
import torch
from src.model.AtrousResidualBlock import AtrousResidualBlock
from ConvLayers import CRLayer, CBNRLayer, CBNDLayer
import math

class MultiResolutionLevel(nn.Module):
    def __init__(self, high_channel=32, low_channel=4, high_depth=3, low_depth=6):
        super().__init__()

        self.sample_count = int(math.log2(high_channel // low_channel))

        self.c_adjust0 = CRLayer(high_channel, low_channel, kernel_size=1, padding=0)
        self.c_adjust1 = CRLayer(low_channel, high_channel, kernel_size=1, padding=0)
        self.c_adjust2 = CRLayer(high_channel*3, high_channel, kernel_size=1, padding=0)

        self.high_sequence = nn.Sequential(
            *[AtrousResidualBlock(channels=high_channel) for _ in range(high_depth)]
        )
        self.low_sequence = nn.Sequential(
            *[AtrousResidualBlock(channels=low_channel) for _ in range(low_depth)]
        )

    def forward(self, input):
        high_input = input
        high_output = self.high_sequence(high_input)

        low_input = self.c_adjust0(input)
        low_output = self.low_sequence(low_input)
        low_output = self.c_adjust1(low_output)

        m_channel_output = torch.cat(
            [high_output, low_output, torch.mul(high_output, low_output)],
            dim=1,
        )

        del high_input, high_output, low_input, low_output  # 释放内存

        return self.c_adjust2(m_channel_output)

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

if __name__ == "__main__":

    input_tensor = torch.randn(1, 32, 200, 200)
    model = MultiResolutionLevel(high_channel=32, low_channel=4, high_depth=3, low_depth=6)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # 应该输出 torch.Size([1, 32+4+32, 200, 200]) 即 torch.Size([1, 68, 200, 200])

    print_model_params(model)
