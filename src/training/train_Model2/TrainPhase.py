import os
import torch
import torch.optim as optim
import torch.utils.data.dataloader as DataLoader
import time
import sys

from src.logging.Logger import Logger
from src.model.Model2.DroneSegModel import DroneSegModel
from src.training.train_Model2.TrainEpoch import train_epoch

def train_phase(
        model: DroneSegModel,
        dataloader: DataLoader,
        logger: Logger,
        epochs: int,
        sample_period: int,
        device: torch.device,
        logging_info: dict,
        criterion: torch.nn.Module,
        optimizer: optim.Optimizer = None,
):
    """
        训练一个阶段（Phase）。

        在每一个 epoch 的训练过程中，模型会统计当前 epoch 的时间、Loss值、MioU值、准确率、
        以及其他相关信息，并将这些信息记录到日志中。根据 sample_period 的设置，模型还会在训练过程中定期采样一些数据进行可视化，并将这些可视化结果保存到日志中。

        训练完成后，模型会将当前阶段的日志进行归档，并保存模型的状态，以便后续分析和使用。
    :param model: 要训练的模型。
    :param dataloader: 数据集
    :param logger: 日志记录器，用于记录训练或评估过程中的信息。
    :param epochs: 训练的 epoch 数量。
    :param sample_period: 表示在训练过程中采样数据进行可视化的频率（例如每多少个 epoch 采样一次）。
    :param device: 设备（CPU 或 GPU），用于将数据和模型移动到适当的计算设备上。
    :param logging_info: 一个包含当前 epoch 和其他相关信息的字典，用于记录日志时使用。
    :param criterion: 损失函数，用于计算模型的损失值。
    :param optimizer: 优化器，用于更新模型的参数（仅在训练阶段使用）。
    :return:
    """

    model.to(device)
    model.train()

    criterion.to(device)

    start_time = time.time()
    for epoch in range(epochs):
        logging_info['epoch'] = epoch + 1  # 更新当前 epoch 信息
        logging_info['elapsed_time'] = time.time() - start_time  # 计算已用时间，单位为秒，并保留4位小数

        print(f"[Epoch {epoch+1}/{epochs}] ", end="")
        sys.stdout.flush()

        train_epoch(
            model=model,
            dataloader=dataloader,
            logger=logger,
            take_sample=(epoch % sample_period == 0),  # 根据 sample_period 决定是否采样
            device=device,
            logging_info=logging_info,
            criterion=criterion,
            optimizer=optimizer,
        )

    model.eval()

    feat_images, labels, images = next(iter(dataloader))
    feat_images = feat_images.to(device)
    outputs = model(feat_images)
    output_labels = torch.argmax(outputs, dim=1)

    logger.save_sample_image(
        output_labels,
        labels.cpu(),
        images.cpu(),
        logging_info="Phase_End"
    )

    del feat_images, labels, images, outputs, output_labels  # 释放内存
