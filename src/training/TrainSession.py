import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.logging.Logger import Logger
from src.training.TrainPhase import train_phase
from typing import Optional
import gc

# default_train_config = [
#     {'lr': 0.1, 'momentum': 0.85, 'epochs': 5},
#
#     {'lr': 0.01, 'momentum': 0.9, 'epochs': 40},
#     {'lr': 0.005, 'momentum': 0.9, 'epochs': 80},
#
#     {'lr': 0.001, 'momentum': 0.9, 'epochs': 80},
#     {'lr': 0.0005, 'momentum': 0.95, 'epochs': 80},
# ]
default_train_config = [
    {'lr': 0.1, 'momentum': 0.85, 'epochs': 5},
    {'lr': 0.01, 'momentum': 0.9, 'epochs': 40},
    {'lr': 0.001, 'momentum': 0.9, 'epochs': 40},
    {'lr': 0.0005, 'momentum': 0.95, 'epochs': 40},
]

# 使用 交叉熵损失函数 和 SGD 优化器 进行训练
def train_session(
        model: nn.Module,
        dataloader: DataLoader,
        criterion,
        title: str,
        device: torch.device,
        train_config: list[dict] = None,
        output_frequency: int = 20,
        label_transform: Optional[callable(torch.Tensor)] = None,
):
    if train_config is None:
        train_config = default_train_config

    print("New training session starting... title: ", title)
    logger = Logger(
        title=title,
        log_format=[
            'phase', 'epoch', 'elapsed_time', 'loss', 'miou', 'accuracy', 'time'
        ],
        output_frequency=output_frequency,
        n_classes=2,
    )

    for phase_idx, (config) in enumerate(train_config):

        optimizer = optim.SGD(
            model.parameters(),
            lr=config['lr'],
            momentum=config['momentum'],
        )

        phase_start_info = f"Phase {phase_idx} Starting | lr: {config['lr']}, momentum: {config['momentum']}, epoch number: {config['epochs']}"
        logger.start_new_phase(f"Phase No.{phase_idx + 1}")
        logger.str_log(phase_start_info)

        train_phase(
            model=model,
            dataloader=dataloader,
            logger=logger,
            epochs=config['epochs'],
            sample_period=min(50, config['epochs'] // 2),  # 每个阶段至少保存 2 次样本
            device=device,
            logging_info={
                'phase': phase_idx + 1,
            },
            criterion=criterion,
            optimizer=optimizer,
            label_transform=label_transform,
        )

        logger.end_current_phase(model_to_save=model)

    logger.finalize_and_plot_all()
    gc.collect()
    torch.cuda.empty_cache()