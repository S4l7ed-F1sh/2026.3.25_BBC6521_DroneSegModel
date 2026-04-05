import torch.nn as nn
from src.model.Model1.AtrousResidualBlock import AtrousResidualBlock

class MultiResolutionLevel(nn.Module):
    def __init__(self, num_blocks, in_c_list, out_c_list):
        super().__init__()

        self.num_blocks = num_blocks

        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(AtrousResidualBlock(
                in_channels=in_c_list[i],
                out_channels=out_c_list[i],
            ))

    def forward(self, level: int, input):
        return self.blocks[level](input)
