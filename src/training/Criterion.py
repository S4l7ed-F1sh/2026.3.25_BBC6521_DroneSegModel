from src.training.TrainBatch import compute_miou
import torch
import torch.nn as nn

# 自定义的损失函数，公式为 Loss = L_{BCE} + L_{Dise} + L_{IoU}
# 输入的参数是模型的输出和标签，模型的输出是一个 B*C*H*W 的张量，标签是一个 B*C*H*W 的 one-hot 编码的张量。
def criterion(output: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    # 计算二元交叉熵损失
    bce_loss = nn.BCEWithLogitsLoss()(output, label)

    # 计算 Dice Loss
    smooth = 1e-6  # 防止除零错误
    output_sigmoid = torch.sigmoid(output)  # 将输出转换为概率
    intersection = (output_sigmoid * label).sum(dim=(2, 3))  # 计算交集
    union = output_sigmoid.sum(dim=(2, 3)) + label.sum(dim=(2, 3))  # 计算并集
    dice_loss = 1 - (2 * intersection + smooth) / (union + smooth)  # 计算 Dice Loss

    # 计算 IoU Loss, 使用提供的 compute_miou 函数计算 IoU，然后转换为 IoU Loss
    output_labels = torch.argmax(output, dim=1)  # 从 (B, C, H, W) 转换为 (B, H, W)
    label_labels = torch.argmax(label, dim=1)  # 从 (B, C, H, W) 转换为 (B, H, W)
    iou_loss = 1 - compute_miou(output_labels.cpu(), label_labels.cpu(), num_classes=output.shape[1])  # 计算 IoU Loss

    # 总损失是三者的加权和，这里权重都设为1，可以根据需要调整
    total_loss = bce_loss + dice_loss.mean() + iou_loss.mean()

    del output, label, output_sigmoid, intersection, union, output_labels, label_labels  # 释放内存

    return total_loss
