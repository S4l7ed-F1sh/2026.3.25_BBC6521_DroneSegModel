import os
from typing import Optional

import torch
import torch.nn as nn
import time

from torch.utils.data import DataLoader

from src.logging.Logger import Logger
from src.training.TrainBatch import train_batch
from src.logging.TimeTransform import format_duration

import gc

def train_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        logger: Logger,
        take_sample: bool,
        device: torch.device,
        logging_info: dict,
        criterion,
        optimizer: torch.optim.Optimizer = None,
        label_transform: Optional[callable(torch.Tensor)] = None,
):
    """
        训练一个 epoch。

        在每一个批次的训练过程中，模型会统计当前批次的时间、Loss值、MioU值、准确率、
    :param model: 要训练的模型。
    :param dataloader: 数据集
    :param logger: 日志记录器，用于记录训练或评估过程中的信息。
    :param take_sample: 表示是否在训练过程中采样一些数据进行可视化的布尔值。
    :param device: 设备（CPU 或 GPU），用于将数据和模型移动到适当的计算设备上。
    :param logging_info: 一个包含当前 epoch 和其他相关信息的字典，用于记录日志时使用。
    :param criterion: 损失函数，用于计算模型的损失值。
    :param optimizer: 优化器，用于更新模型的参数（仅在训练阶段使用）。
    :param label_transform: 可选的标签变换函数，用于在训练前对标签进行预处理。
    :return:
    """
    model.train()

    batch_numbers = len(dataloader)
    total_loss = 0.0
    total_miou = 0.0
    total_accuracy = 0.0
    start_time = time.time()

    with torch.set_grad_enabled(True):
        for batch_idx, (feat_images, labels, images) in enumerate(dataloader):

            feat_images, labels, images = feat_images.to(device), labels.to(device), images.to(device)

            if label_transform is not None:
                labels = label_transform(labels)

            batch_loss, batch_miou, batch_accuracy = train_batch(
                model=model,
                batch_img=feat_images,
                batch_lbl=labels,
                criterion=criterion,
                optimizer=optimizer,
            )

            total_loss += batch_loss
            total_miou += batch_miou
            total_accuracy += batch_accuracy

    total_time = time.time() - start_time  # 计算总时间，单位为秒

    avg_miou = total_miou / batch_numbers
    avg_accuracy = total_accuracy / batch_numbers

    logging_info.update({
        'elapsed_time': format_duration(logging_info['elapsed_time'] + total_time),  # 累加已用时间
        'loss': total_loss,
        'miou': avg_miou,
        'accuracy': avg_accuracy,
        'time': format_duration(total_time),
    })
    logger.log(log_data=logging_info)

    if take_sample:
        model.eval()

        with torch.no_grad():

            feat_images, labels, images = next(iter(dataloader))
            feat_images = feat_images.to(device)

            outputs = model(feat_images)
            output_labels = torch.argmax(outputs, dim=1).detach().cpu()

            logger.save_sample_image(
                output_labels,
                labels,
                images,
                logging_info=f"E{logging_info['epoch']}"
            )

            del feat_images, labels, images, outputs, output_labels  # 释放内存

    del total_loss, total_miou, total_accuracy  # 释放内存

    gc.collect()
    torch.cuda.empty_cache()
