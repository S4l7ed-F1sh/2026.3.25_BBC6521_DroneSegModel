import os
import torch
import torch.nn as nn

def compute_miou(preds, labels, num_classes=5):
    """
    计算 Mean Intersection over Union (mIoU) 指标。

    Args:
        preds (torch.Tensor): 模型的预测结果，形状为 (batch_size, height, width)，每个元素是预测的类别索引。
        labels (torch.Tensor): 真实的标签，形状为 (batch_size, height, width)，每个元素是实际的类别索引。
        num_classes (int): 类别的数量，默认为 5。
    """
    ious = []
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        label_inds = (labels == cls)

        intersection = (pred_inds & label_inds).sum().item()
        union = (pred_inds | label_inds).sum().item()

        if union == 0:
            iou = float('nan')  # 如果没有该类别的像素，IoU 定义为 NaN
        else:
            iou = intersection / union

        ious.append(iou)

    miou = torch.tensor(ious).nanmean().item()  # 计算 mIoU，忽略 NaN 值
    return miou

def train_batch(
        model: nn.Module,
        batch_img: torch.Tensor,
        batch_lbl: torch.Tensor,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
):
    """
        训练一个批次。
    :param model: 要训练的模型。
    :param batch_img: 输入的图像数据，形状为 (B, C, H, W)。
    :param batch_lbl: 输入的标签数据，形状为 (B, H, W)。
    :param criterion: 损失函数，用于计算模型的损失值。
    :param optimizer: 优化器，用于更新模型的参数。
    :return:
    """

    model.train()

    optimizer.zero_grad()

    outputs = model(batch_img)

    loss = criterion(outputs, batch_lbl)

    loss.backward()
    optimizer.step()

    n_classes = outputs.shape[1]  # 获取类别数量

    if len(batch_lbl.shape) == 4:
        if (batch_lbl.shape[1] == 1):
            batch_lbl = batch_lbl.squeeze(1)  # 从 (B, 1, H, W) 转换为 (B, H, W)
        else:
            batch_lbl = torch.argmax(batch_lbl, dim=1)  # 从 (B, C, H, W) 转换为 (B, H, W)
    batch_lbl = batch_lbl.detach().cpu()  # 将标签移动到 CPU 上进行计算
    outputs = torch.argmax(outputs, dim=1).cpu()  # 转换为 (B, H, W)，并移动到 CPU 上进行计算

    batch_miou = compute_miou(outputs, batch_lbl, num_classes=n_classes)
    batch_accuracy = (outputs == batch_lbl).float().mean().item()

    del batch_img, batch_lbl, outputs  # 释放内存

    return loss.item(), batch_miou, batch_accuracy
