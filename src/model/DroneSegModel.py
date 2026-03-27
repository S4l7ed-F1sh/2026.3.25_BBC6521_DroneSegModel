import os
import torch
import torch.nn as nn
from src.model.AtrousResidualBlock import AtrousResidualBlock
from src.model.MultiResolutionLevel import MultiResolutionLevel
from src.model.ConvAndReluLayer import CRLayer, CBNRLayer

# 输入的图片有 32 个通道，输出的图片有 n_classes 个通道
# 模型的结构设计思路：分为 layers 层 MultiResolutionLevel，每层的最大通道数为 max_channels。
# 第 i 层的输入通道为 input_channels // (2**i)，输出通道为 max_channels。分辨率为输入图片分辨率的 (2**i) 分之一。
class DroneSegModel(torch.nn.Module):
    def __init__(self, n_classes=5, layers=4, max_channels=128, input_channels=32, feat_channels=22):
        super(DroneSegModel, self).__init__()

        self.n_classes = n_classes
        self.layers = layers
        self.max_channels = max_channels
        self.input_channels = input_channels
        self.feat_channels = feat_channels

        self.f2i_conv1 = CBNRLayer(in_channels=feat_channels, out_channels=input_channels*2, kernel_size=3, padding=1)
        self.f2i_conv2 = CBNRLayer(in_channels=input_channels*2, out_channels=input_channels, kernel_size=3, padding=1)

        # 每层每个模块的输入输出通道数比较复杂，用两个二维表动态计算一下
        self.in_c = [[0] * (layers + 3) for _ in range(layers)]
        self.out_c = [[0] * (layers + 3) for _ in range(layers)]
        # 每个模块的输出通道数为 out_channels = min(in_channels, max_channels)。
        # 在不超过 max_channels 的情况下不会改变通道数。如果输入通道数超过则会进行压缩。
        self.in_c[0][0] = input_channels
        self.out_c[0][0] = min(self.in_c[0][0], max_channels)

        # 先计算每层第一个模块的通道数，每层第一个模块的输入通道数为 input_channels // (2**i)，输出通道数为 min(input_channels // (2**i), max_channels)
        cur_c = input_channels
        self.start_convs = nn.ModuleList()
        for i in range(layers - 1):
            cur_c = cur_c // 2

            self.start_convs.append(CRLayer(
                in_channels=input_channels,
                out_channels=cur_c,
                kernel_size=3,
                padding=1,
            ))
            self.in_c[i+1][0] = cur_c
            self.out_c[i+1][0] = min(cur_c, max_channels)

        # 接下来计算后续模块的通道数。
        # 第 i 层的模块数量为 i+3。
        # 从每层第二个模块开始，每个模块的输入通道数为前一个模块的输出通道数加上上一层同位置模块的输出通道数（如果存在）加上上一层相邻位置模块的输出通道数（如果存在）。输出通道数为输入通道数和 max_channels 中的较小值。
        for j in range(layers+2):
            for i in range(layers):
                self.in_c[i][j+1] = self.out_c[i][j]
                if i != 0:
                    self.in_c[i][j+1] += self.out_c[i-1][j]
                if i != layers-1:
                    self.in_c[i][j+1] += self.out_c[i+1][j]

                self.out_c[i][j+1] = min(self.in_c[i][j+1], max_channels)

        # 输出计算出来的 in_c 和 out_c 表格，方便后续调试
        print("输入通道数表 (in_c):")
        for i in range(layers):
            print(f"层 {i}: {self.in_c[i][:layers+3]}")
        print("\n输出通道数表 (out_c):")
        for i in range(layers):
            print(f"层 {i}: {self.out_c[i][:layers+3]}")

        # 每层最后一个模块的输出通道有可能低于 max_channels，所以需要一个 1*1 的卷积将通道数调整到 max_channels，方便后续的特征融合。
        self.end_convs = nn.ModuleList()
        for i in range(layers):
            self.end_convs.append(CBNRLayer(
                in_channels=self.out_c[i][i+2],
                out_channels=max_channels,
                kernel_size=3,
                padding=1,
            ))

        # 初始化 MultiResolutionLevel 模块，每层的模块数量为 i+3，每个模块的输入输出通道数根据前面计算的 in_c 和 out_c 表来设置。
        self.multi_res_levels = nn.ModuleList()
        for i in range(layers):
            self.multi_res_levels.append(MultiResolutionLevel(
                num_blocks=i+3,
                in_c_list=self.in_c[i][:],
                out_c_list=self.out_c[i][:],
            ))

        # 在训练中，最后的多层输出会进行拼接，然后通过两个 1*1 的卷积调整通道数，得到最终的输出特征图。
        self.final_conv1 = CBNRLayer(
            in_channels=max_channels*layers,
            out_channels=max_channels,
            kernel_size=1,
            padding=0,
        )
        self.final_conv2 = CBNRLayer(
            in_channels=max_channels,
            out_channels=n_classes,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x, mode='train'):
        assert mode in ['pretrain', 'train'], "mode must be 'pretrain' or 'train'"

        x = self.f2i_conv1(x)
        x = self.f2i_conv2(x)

        if mode == 'pretrain':
            # 预训练模块随机启用其中一个层
            l = torch.randint(0, self.layers, size=()).item()

            # 先对输入数据进行下采样，使用平均池化，得到不同分辨率的输入特征图
            for i in range(l):
                x = nn.AvgPool2d(kernel_size=2, stride=2)(x)

            # 如果不是第一层，需要使用 start_convs 将输入通道数调整到当前层的输入通道数
            if l != 0:
                x = self.start_convs[l-1](x)

            # 按顺序通过当前层的模块进行前向传播，不需要上下两层特征拼接
            # 在只启用单层的时候，因为无法拼接前一层和后一层的特征，所以必须以 0 填充前一层和后一层的特征，这样才能保证输入通道数的一致性。
            for j in range(l+3):
                x = self.multi_res_levels[l].forward(
                    j,
                    torch.cat(
                        [
                            x,
                            torch.zeros(x.shape[0], self.in_c[l][j] - x.shape[1], x.shape[2], x.shape[3], device=x.device)
                        ],
                        dim=1,
                    )
                )

            # 完成后使用 end_convs 将通道数调整到 max_channels
            x = self.end_convs[l](x)

            # 处理完成，经过多次线性插值上采样，恢复到输入图片的分辨率
            for i in range(l):
                x = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)(x)

            # 最后使用 final_conv2 将通道数调整到 n_classes，得到最终的输出特征图
            x = self.final_conv2(x)
            return x
        else:
            # 正式训练模块，使用平均池化得到不同分辨率的输入特征图
            down_sample = [x]
            for i in range(self.layers-1):
                down_sample.append(nn.AvgPool2d(kernel_size=2, stride=2)(down_sample[i]))

            # 使用 start_convs 将除了第一层以外的输入通道数调整到当前层的输入通道数
            inputs = [x]
            for i in range(self.layers-1):
                inputs.append(self.start_convs[i](down_sample[i+1]))

            # 按顺序通过每层的模块进行前向传播，每个模块的输入特征图为当前层前一个模块的输出特征图与上一层同位置模块的输出特征图（如果存在）与上一层相邻位置模块的输出特征图（如果存在）进行拼接得到的特征图
            nxt_lvl = []
            for i in range(self.layers):
                nxt_lvl.append(self.multi_res_levels[i].forward(0, inputs[i]))

            for i in range(self.layers):
                if i == 0:
                    inputs[i] = torch.cat(
                        [
                            nxt_lvl[0],
                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)(nxt_lvl[1]),
                        ],
                        dim=1
                    )
                elif i == self.layers-1:
                    inputs[i] = torch.cat(
                        [
                            nxt_lvl[i],
                            nn.AvgPool2d(kernel_size=2, stride=2)(nxt_lvl[i-1]),
                        ],
                        dim=1
                    )
                else:
                    inputs[i] = torch.cat(
                        [
                            nxt_lvl[i],
                            nn.AvgPool2d(kernel_size=2, stride=2)(nxt_lvl[i-1]),
                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)(nxt_lvl[i+1]),
                        ],
                        dim=1
                    )

            # 用来存放每层最后一个阶段的输出特征图，后续会通过 end_convs 将通道数调整到 max_channels，方便后续的特征融合。
            final_features = []

            # min_layer 表示当前阶段处理的最低层数，注意到第一层只有 3 个阶段，之后每一层都会增加一个阶段，所以每处理完一层就将 min_layer 加 1，这样就能保证每层的模块在正确的阶段进行前向传播。
            min_layer = 0
            for j in range(self.layers+1):
                # 计算本阶段各层的输出特征图，注意只有大于等于 min_layer 的层会进行前向传播
                for i in range(min_layer, self.layers):
                    nxt_lvl[i] = self.multi_res_levels[i].forward(j+1, inputs[i])

                # 当 j>0 时，说明已经处理完了第一层的所有阶段，接下来每处理完一层就将 min_layer 加 1，这样就能保证每层的模块在正确的阶段进行前向传播。
                # 同时应该将已经处理完的层的最后一个阶段的输出特征图通过 end_convs 将通道数调整到 max_channels，方便后续的特征融合。
                if j > 0:
                    new_tensor = nxt_lvl[min_layer]
                    for k in range(min_layer):
                        new_tensor = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)(new_tensor)
                    final_features.append(self.end_convs[min_layer](new_tensor))
                    min_layer += 1

                # 计算下一阶段的输入特征图，每个模块的输入特征图为当前层前一个模块的输出特征图与上一层同位置模块的输出特征图（如果存在）与上一层相邻位置模块的输出特征图（如果存在）进行拼接得到的特征图
                for i in range(min_layer, self.layers):
                    if i == 0:
                        inputs[i] = torch.cat(
                            [
                                nxt_lvl[0],
                                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)(nxt_lvl[1])
                            ],
                            dim=1
                        )
                    elif i == self.layers-1:
                        inputs[i] = torch.cat(
                            [
                                nxt_lvl[i],
                                nn.AvgPool2d(kernel_size=2, stride=2)(nxt_lvl[i-1]),
                            ],
                            dim=1
                        )
                    else:
                        inputs[i] = torch.cat(
                            [
                                nxt_lvl[i],
                                nn.AvgPool2d(kernel_size=2, stride=2)(nxt_lvl[i-1]),
                                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)(nxt_lvl[i+1]),
                            ],
                            dim=1
                        )

            # 最后，将每层最后一个阶段的输出特征图进行拼接，然后通过两个 1*1 的卷积调整通道数，得到最终的输出特征图。
            final_tensor = torch.cat(final_features, dim=1)
            final_tensor = self.final_conv1(final_tensor)
            final_tensor = self.final_conv2(final_tensor)

            return final_tensor

if __name__ == "__main__":
    model = DroneSegModel(n_classes=5, layers=4, max_channels=64, input_channels=32)
    x = torch.randn(1, 22, 64, 64)  # 假设输入大小为 1x32x64x64
    output = model(x, mode='train')
    print(output.shape)  # 应输出类似 torch.Size([1, 5, 64, 64]) 的结果


    x = torch.randn(1, 22, 64, 64)  # 假设输入大小为 1x32x64x64
    output = model(x, mode='pretrain')
    print(output.shape)  # 应输出类似 torch.Size([1, 5, 64, 64]) 的结果
