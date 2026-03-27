import os
import csv
import sys

from PIL import Image
import matplotlib.pyplot as plt
from PIL import Image

# PhaseLogger 用于记录阶段的日志信息，在构造后将会传入一个阶段名称和 Logger 的基础目录路径，
# 还有一个输出频率，表示每几轮日志，PhaseLogger 会用 print 输出一行控制台报告，以及日志 dict 键值内容，因为 PhaseLogger 会保存日志信息，这里的 format 表示有哪些列。
# PhaseLogger 会在基础目录下创建一个以阶段名称命名的子目录，并将日志信息保存在该子目录下。
# 在子目录下，PhaseLogger 会创建一个名为 saving_log 的 txt 文件，用于记录日志信息。同时 PhaseLogger 自己也会维护一个 saving_log 列表，用于在内存中保存日志信息，方便后续处理。
# PhaseLogger 拥有一个 log 方法，用于记录日志信息，方法的传入参数为 dict 类型，包含日志信息的键值对。
# 键值对的格式是固定的，PhaseLogger 会将日志信息转换为字符串，输入到 saving_log.txt 中，并且在内存中保存到 saving_log 列表中。
# 当日志信息的轮数达到输出频率时，PhaseLogger 会用 print 输出一行控制台报告，报告的内容包括当前阶段名称、轮数以及日志信息的键值内容。
# 同时，PhaseLogger 还有一个 str_log 方法，接受一个字符串类型的日志信息，这个方法用于特殊情况如出现错误的时候，会将日志信息输入到 saving_log.txt 和控制台。
# 在阶段结束时，PhaseLogger 会将保存的日志进行归档，然后创建一个 saveing_log.csv 的表格，将日志信息保存到表格中，方便后续分析和使用。
# 并且，PhaseLogger 还会将日志信息进行可视化，用 matplotlib 绘制训练曲线图，保存到 logs/训练子阶段 目录下。
# 最后，PhaseLogger 会提供 getter 方法，获取保存的日志信息，方便后续分析和使用。
class PhaseLogger:
    """
    PhaseLogger 用于记录阶段的日志信息。

    Args:
        phase_name (str): 阶段的名称。
        base_dir (str): 日志文件的基础存储目录。
        log_format (list): 日志的格式，定义了日志字典中应包含的键名列表。
        output_frequency (int): 控制台输出的频率，默认为10。
    """

    def __init__(self, phase_name, base_dir, log_format, color_list, output_frequency=10):
        """
        初始化 PhaseLogger。

        Args:
            phase_name (str): 阶段的名称。
            base_dir (str): 日志文件的基础存储目录。
            log_format (list): 日志的格式，例如 ['epoch', 'loss', 'accuracy']。
            output_frequency (int): 控制台输出的频率，默认为10。
            color_list (list): 包含了 n_classes 个颜色，用于渲染采样分割图。
        """
        self.phase_name = phase_name
        self.base_dir = os.path.join(base_dir, self.phase_name)
        self.output_frequency = output_frequency

        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        self.log_format = log_format
        self.saving_log = []
        self._log_file_path = os.path.join(self.base_dir, "saving_log.txt")
        # 清空或创建新的日志文件
        open(self._log_file_path, 'w').close()

        self.color_list = color_list  # 存储颜色映射字典，供后续使用

    def format_log_value(self, key, value):
        """
        根据键名对值进行格式化
        """
        # 定义需要特殊格式化的键及其格式
        float_formats = {
            'elapsed_time': '.4f',  # 时间保留4位小数
            'loss': '.4f',  # 损失保留4位小数
            'miou': '.4f',  # mIoU保留4位小数
            'accuracy': '.4f'  # 准确率保留4位小数
        }

        # 如果该键需要特殊格式化，则应用它
        if key in float_formats:
            return f"{value:{float_formats[key]}}"
        else:
            # 否则返回原始值的字符串形式
            return str(value)

    def log(self, log_data):
        """
        记录一条日志信息。

        Args:
            log_data (dict): 包含日志信息的字典，其键必须与 log_format 定义的相符。
        """
        if not isinstance(log_data, dict):
            raise TypeError("log_data must be a dictionary.")

        # 验证 log_data 的键是否与 log_format 一致
        if set(log_data.keys()) != set(self.log_format):
            raise ValueError(f"log_data keys {set(log_data.keys())} do not match log_format {set(self.log_format)}")

        self.saving_log.append(log_data)

        output_str = f"[Phase {self.phase_name}]---" + ' | '.join([f"{key}: {self.format_log_value(key=key, value=log_data[key])}" for key in self.log_format])

        with open(self._log_file_path, 'a') as f:
            f.write(output_str + '\n')

        # 如果达到输出频率，则打印报告
        current_round = len(self.saving_log)
        if current_round % self.output_frequency == 0:
            self._print_report(output_str)

    def str_log(self, message):
        """
        记录一条字符串类型的日志信息（例如错误信息）。

        Args:
            message (str): 要记录的字符串消息。
        """
        timestamped_message = f"[{self.phase_name}] {message}"

        with open(self._log_file_path, 'a', encoding='utf-8') as f:
            f.write(timestamped_message + '\n')
        print(timestamped_message)
        sys.stdout.flush()  # 确保日志立即输出到控制台

    def _print_report(self, report_content):
        """
        内部方法，用于打印控制台报告。

        Args:
            report_content (str): 要打印的报告内容。
        """

        print(report_content)
        sys.stdout.flush()  # 确保日志立即输出到控制台

    def end_phase(self):
        """
        结束当前阶段，将日志归档为 CSV 文件并生成可视化图表。
        """
        # 1. 创建 CSV 文件
        csv_filename = os.path.join(self.base_dir, "saving_log.csv")
        if self.saving_log:
            with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.log_format)
                writer.writeheader()
                for log_entry in self.saving_log:
                    writer.writerow(log_entry)

        # 2. 生成可视化图表
        if self.saving_log:
            self._plot_logs()

    def _plot_logs(self):
        """
        内部方法，使用 matplotlib 绘制日志图表。
        """
        num_plots = len(self.log_format)
        if num_plots <= 0:
            return

        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots))
        if num_plots == 1:
            axes = [axes]

        rounds = list(range(1, len(self.saving_log) + 1))

        for i, key in enumerate(self.log_format):
            values = [entry[key] for entry in self.saving_log]
            axes[i].plot(rounds, values, marker='o', linestyle='-', label=key)
            axes[i].set_title(f'{self.phase_name} - {key}')
            axes[i].set_xlabel('Round')
            axes[i].set_ylabel(key)
            axes[i].grid(True)
            axes[i].legend()

        plt.tight_layout()
        plot_path = os.path.join(self.base_dir, f"{self.phase_name}_plot.png")
        plt.savefig(plot_path)
        plt.close()  # 关闭图形以释放内存

    def save_sample_image(self, out_img, lbl_img, img, logging_info):
        """
        采样保存图片，第一个是模型输出的分割结果图，第二个是标签图，第三个是原始图像。
        :param out_img: 模型输出的结果分割图，为多通道张量。
        :param lbl_img: 标签图，为多通道张量。
        :param img: 原始图像，为多通道张量，被归一化至 0 到 1 之间。
        :param logging_info: 在保存图片时在名称中添加一些日志信息，例如 epoch 轮数等，方便后续分析。
        :return:
        """

        # 确保张量在 CPU 上，并转换为 NumPy 数组
        out_img_np = out_img.cpu().numpy()
        lbl_img_np = lbl_img.cpu().numpy()
        img_np = img.cpu().numpy()

        for i in range(out_img.shape[0]):
            # 将 out_img 和 lbl_img 转换为 PNG 格式的 P 模式图像
            out_pil = Image.fromarray(out_img_np[i].astype('uint8'), mode='P')
            lbl_pil = Image.fromarray(lbl_img_np[i].astype('uint8'), mode='P')

            # 设置调色板
            out_pil.putpalette(self.color_list)
            lbl_pil.putpalette(self.color_list)

            # 将原始图像转换为 RGB 模式的 PIL 图像
            img_pil = Image.fromarray((img_np[i] * 255).clip(0, 255).astype('uint8'), mode='RGB')

            # 创建保存目录（如果不存在）
            samples_dir = os.path.join(self.base_dir, "samples")
            os.makedirs(samples_dir, exist_ok=True)

            # 保存图片
            out_pil.save(os.path.join(samples_dir, f"out_{logging_info}_bid{i}_sample.png"))
            lbl_pil.save(os.path.join(samples_dir, f"lbl_{logging_info}_bid{i}_sample.png"))
            img_pil.save(os.path.join(samples_dir, f"img_{logging_info}_bid{i}_sample.png"))

    def get_saving_log(self):
        """
        Getter 方法，返回保存的日志列表。

        Returns:
            list: 一个包含所有日志字典的列表。
        """
        return self.saving_log
