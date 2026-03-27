import os
import matplotlib.pyplot as plt
from src.logging.PhaseLogger import PhaseLogger
import numpy as np
import sys

# Logger 用于模型训练过程中的日志记录，支持输出到控制台和文件。
# 日志保存在 resources/logs/ 目录下，目录下有一个名为 title 的子目录，子目录下按照训练的阶段创建一系列子目录。
# 每个训练阶段对应一个 PhaseLogger 对象，PhaseLogger 用于记录该阶段的日志信息，包括训练进度、损失值、评估指标等。
# Logger 提供 log, str_log 等方法，当当前有正在进行的训练阶段时，调用这些方法会将日志信息传递给当前阶段的 PhaseLogger 对象进行记录和输出。
# 在每个训练阶段结束时，PhaseLogger 会将保存的日志进行归档，创建一个 saving_log.csv 的表格，将日志信息保存到表格中，并且生成图表，方便后续分析和使用。
# 除此以外，Logger 提供一个保存模型参数的功能，在阶段结束的时候会传入类型的对象，这个时候会将模型的参数保存至 resources/logs/title/子阶段名称/archive 目录下。
# Logger 还提供一个最终的方法，在调用时要求当前训练阶段已经结束，这个时候 Logger 会遍历已有的 PhaseLogger 对象，获取每个阶段的日志信息，
# 整合并且用 matplotlib 绘制包含所有阶段训练信息的图表，保存至 resources/logs/title/ 目录下。
class Logger:
    """
    Logger 用于模型训练过程中的日志记录，支持输出到控制台和文件。
    它管理多个 PhaseLogger 实例，代表不同的训练阶段。

    """

    def __init__(self, title, log_format, output_frequency=10, n_classes=5):
        """
        初始化 Logger。

        Args:
            title (str): 训练任务的标题，用于创建日志目录。
            log_format (list): 日志的格式，定义了日志字典中应包含的键名列表。
            output_frequency (int): 控制台输出的频率，默认为10。
            n_classes(int): 语义分割模型的类别数量，默认为5，在采样保存分割图片时，需要生成对应数量的颜色来区分不同类别的分割结果。
        """
        self.title = title
        self.log_format = log_format
        self.output_frequency = output_frequency

        self.base_dir = os.path.join('resources/logs', self.title)
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        self.phase_loggers = []  # 用于存储不同阶段的 PhaseLogger 对象
        self.current_phase_logger = None  # 当前活跃的 PhaseLogger

        # 随机生成 n_classes 个颜色，用于分割结果的可视化
        # 1. 获取颜色 (0-1 浮点数)
        cmap = plt.cm.get_cmap('tab10', n_classes)
        colors_float = cmap.colors

        # 2. 转换为 0-255 整数，并展平列表
        # 将 [(r,g,b), ...] 转换为 [r, g, b, r, g, b, ...] 且数值为 int
        color_list_int = (np.array(colors_float) * 255).astype(int).flatten().tolist()

        # 3. 补全调色板 (可选但推荐)
        # PIL 的 P 模式调色板通常需要 768 个值 (256个颜色 * 3通道)
        # 如果 n_classes < 256，剩下的用 0 填充
        if len(color_list_int) < 768:
            color_list_int += [0] * (768 - len(color_list_int))

        self.color_list = color_list_int
        # --- 结束 ---

    def start_new_phase(self, phase_name):
        """
        开始一个新的训练阶段。

        Args:
            phase_name (str): 新阶段的名称。
        """
        if self.current_phase_logger is not None:
            raise RuntimeError(
                f"A phase named '{self.current_phase_logger.phase_name}' is already in progress. "
                f"Please call 'end_current_phase' before starting a new one."
            )

        new_logger = PhaseLogger(phase_name, self.base_dir, self.log_format, self.color_list, self.output_frequency)
        self.phase_loggers.append(new_logger)
        self.current_phase_logger = new_logger
        print(f"Started new logging phase: '{phase_name}'")
        sys.stdout.flush()  # 确保日志立即输出到控制台

    def end_current_phase(self, model_to_save=None):
        """
        结束当前的训练阶段，并可选择性地保存模型。

        Args:
            model_to_save (object, optional): 一个具有 save 方法的对象（如 PyTorch 模型），
                                              其参数将被保存到 archive 目录。
        """
        if self.current_phase_logger is None:
            raise RuntimeError("No active phase to end.")

        # 1. 结束当前 PhaseLogger 的工作（生成CSV和图表）
        self.current_phase_logger.end_phase()

        # 2. 保存模型（如果提供了模型）
        if model_to_save is not None:
            archive_dir = os.path.join(self.current_phase_logger.base_dir, 'archive')
            if not os.path.exists(archive_dir):
                os.makedirs(archive_dir)

            model_save_path = os.path.join(archive_dir, f"{self.current_phase_logger.phase_name}_model.pth")
            # 假设 model_to_save 是一个具有 state_dict() 方法的 PyTorch 模型
            # 或者任何拥有 save(path) 方法的对象
            if hasattr(model_to_save, 'state_dict'):
                import torch
                torch.save(model_to_save.state_dict(), model_save_path)
            elif hasattr(model_to_save, 'save'):
                model_to_save.save(model_save_path)
            else:
                # 如果都不是，则尝试直接序列化整个对象
                import pickle
                with open(model_save_path, 'wb') as f:
                    pickle.dump(model_to_save, f)
            print(f"Model saved to {model_save_path}")
            sys.stdout.flush()  # 确保日志立即输出到控制台

        # 3. 将当前 PhaseLogger 置为空，准备下一个阶段
        self.current_phase_logger = None

    def log(self, log_data):
        """
        记录一条日志信息。此方法会将日志传递给当前活跃的 PhaseLogger。

        Args:
            log_data (dict): 包含日志信息的字典。
        """
        if self.current_phase_logger is None:
            raise RuntimeError("No active phase. Please start a phase first using 'start_new_phase'.")
        self.current_phase_logger.log(log_data)

    def str_log(self, message):
        """
        记录一条字符串类型的日志信息。此方法会将日志传递给当前活跃的 PhaseLogger。

        Args:
            message (str): 要记录的字符串消息。
        """
        if self.current_phase_logger is None:
            raise RuntimeError("No active phase. Please start a phase first using 'start_new_phase'.")
        self.current_phase_logger.str_log(message)

    def save_sample_image(self, output, label, image, logging_info):
        """
        保存一个样本的分割结果图像。

        Args:
            output (torch.Tensor): 模型的输出张量，形状为 (B, 1, H, W)，其中 C 是类别数量。
            label (torch.Tensor): 真实标签张量，形状为 (B, H, W)。
            image (torch.Tensor): 原始输入图像张量，形状为 (3, H, W)。
            logging_info (str): 包含日志信息的字符串，用于生成文件名。
        """
        if self.current_phase_logger is None:
            raise RuntimeError("No active phase. Please start a phase first using 'start_new_phase'.")

        self.current_phase_logger.save_sample_image(output, label, image, logging_info)

    def finalize_and_plot_all(self):
        """
        在所有阶段结束后，整合所有阶段的日志并绘制总览图表。
        """
        if self.current_phase_logger is not None:
            raise RuntimeError(
                "A phase is still in progress ('{}'). "
                "Please call 'end_current_phase' for all started phases before finalizing.".format(
                    self.current_phase_logger.phase_name
                )
            )

        if not self.phase_loggers:
            print("No phases were logged. Skipping final plot generation.")
            sys.stdout.flush()  # 确保日志立即输出到控制台
            return

        # 为每种日志项创建一个总的图表
        all_keys = set()
        for logger in self.phase_loggers:
            all_keys.update(logger.log_format)

        num_plots = len(all_keys)
        if num_plots == 0:
            return

        fig, axes = plt.subplots(num_plots, 1, figsize=(12, 5 * num_plots))
        if num_plots == 1:
            axes = [axes]

        for ax, key in zip(axes, all_keys):
            global_round_offset = 0
            for logger in self.phase_loggers:
                phase_logs = logger.get_saving_log()
                if key in logger.log_format and phase_logs:
                    values = [entry[key] for entry in phase_logs]
                    rounds = list(range(global_round_offset, global_round_offset + len(values)))

                    ax.plot(rounds, values, marker='o', linestyle='-', label=f'{logger.phase_name}', alpha=0.7)
                    ax.axvline(x=global_round_offset, color='gray', linestyle='--', alpha=0.5)  # 标记阶段分割线

                    global_round_offset += len(values)

            ax.set_title(f'All Phases - {key}')
            ax.set_xlabel('Global Round')
            ax.set_ylabel(key)
            ax.legend()
            ax.grid(True)

        plt.tight_layout()

        final_plot_path = os.path.join(self.base_dir, f"{self.title}_all_phases_summary.png")
        plt.savefig(final_plot_path)
        plt.close()
        print(f"Final summary plot saved to {final_plot_path}")
        sys.stdout.flush()  # 确保日志立即输出到控制台