import os
import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader

from src.logging.Logger import Logger
from src.model.DroneSegModel import DroneSegModel

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

def train_epoch(
        model: DroneSegModel,
        dataloader: DataLoader,
        logger: Logger,
        is_pretrain: bool,
        take_sample: bool,
        device: torch.device,
        logging_info: dict,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer = None,
):
    """
        训练一个 epoch。

        在每一个批次的训练过程中，模型会统计当前批次的时间、Loss值、MioU值、准确率、
    :param model: 要训练的模型。
    :param dataloader: 数据集
    :param logger: 日志记录器，用于记录训练或评估过程中的信息。
    :param is_pretrain: 表示当前阶段是否是预训练阶段的布尔值。
    :param take_sample: 表示是否在训练过程中采样一些数据进行可视化的布尔值。
    :param device: 设备（CPU 或 GPU），用于将数据和模型移动到适当的计算设备上。
    :param logging_info: 一个包含当前 epoch 和其他相关信息的字典，用于记录日志时使用。
    :param criterion: 损失函数，用于计算模型的损失值。
    :param optimizer: 优化器，用于更新模型的参数（仅在训练阶段使用）。
    :return:
    """
    # 在传入的时候保证模型已经被移动到正确的设备上，这里再确认一下
    model.to(device)
    model.train()

    batch_numbers = len(dataloader)
    total_loss = 0.0
    total_miou = 0.0
    total_accuracy = 0.0
    start_time = time.time()

    with torch.set_grad_enabled(True):
        for batch_idx, (feat_images, labels, images) in enumerate(dataloader):

            optimizer.zero_grad()
            feat_images = feat_images.to(device)
            labels = labels.to(device)

            outputs = model(feat_images, mode='pretrain' if is_pretrain else 'train')
            output_labels = torch.argmax(outputs, dim=1)

            loss = criterion(outputs, labels)

            total_loss += loss.item()
            total_miou += compute_miou(output_labels, labels)
            total_accuracy += (output_labels == labels).float().mean().item()

            loss.backward()
            optimizer.step()

    total_time = time.time() - start_time

    avg_miou = total_miou / batch_numbers
    avg_accuracy = total_accuracy / batch_numbers

    logging_info.update({
        'loss': total_loss,
        'miou': avg_miou,
        'accuracy': avg_accuracy,
        'time': total_time,
    })
    logger.log(log_data=logging_info)

    if take_sample:
        model.eval()

        feat_images, labels, images = next(iter(dataloader))
        feat_images = feat_images.to(device)
        outputs = model(feat_images, mode='pretrain' if is_pretrain else 'train')
        output_labels = torch.argmax(outputs, dim=1)

        logger.save_sample_image(
            output_labels,
            labels.cpu(),
            images.cpu(),
            logging_info=f"E{logging_info['epoch']}"
        )

        del feat_images, labels, images, outputs, output_labels  # 释放内存

    del total_loss, total_miou, total_accuracy  # 释放内存
