import os
import csv
import time
import torch
import platform
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# 导入tqdm用于显示进度条
try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm not found. Please install it with 'pip install tqdm' for better progress visualization.")
    tqdm = None


# 本文件定义了一个函数用于模型性能评估。
# dataloader 为数据加载器，batch_size 为 1
# dataloader 提供一个三元组，为 feature_input, label, image 的形式
# 其中 feature_input 为处理过的输入，通道数为 22，label 为标签，通道数为 1。image 为原始图像，通道数为 3。
# image 不会用到。
# 模型输出的图像如果为 n_classes 个通道，则需要对输出进行 argmax 操作，得到每个像素的类别索引。这个操作在评估指标计算中是必要的，因为指标通常需要离散的类别标签。
# 运行一个 epoch，对于每个 batch，在获得模型输出后，将模型输出和 label 都处理成 1 通道的形式，值域为整数的 [0, nclasses-1]
# 函数需要收集的是 mIoU, pixel accuracy, class-wise accuracy, confusion matrix, 以及每个类别的 precision 和 recall，以及最后的 F1 score。
# 还需要收集 total_time, time_per_frame，还有最大显存占用、硬件信息、框架（pytorch）的版本这些信息。
# 首先，函数需要打开 resources/benchmark/{title}/ 目录，然后创建一个 csv 文件用于记录该模型的指标，不用记录每个 batch 的指标，只记录每个 epoch 的平均指标。
# 同时，需要将这些信息输出至控制台，以及写入同目录下新建的 log.txt 文件中。
def benchmark(
        title: str,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        need_argmax: bool,
        device: torch.device,
        need_featured_input: bool,
        n_classes: int = 5,
):
    """
    Benchmark a semantic segmentation model on a given dataloader.

    Args:
        title (str): Name of the experiment; used for folder and file naming.
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader providing (feature_input, label, image).
                                 feature_input: [B, 22, H, W], label: [B, 1, H, W]
        need_argmax (bool): Whether to apply argmax on model output.
        device (torch.device): Device to run inference on.
        n_classes (int): Number of semantic classes.
    """
    print(f"Starting benchmark for: {title}")
    print(f"Number of classes: {n_classes}")
    print(f"Device: {device}")
    print(f"Need argmax: {need_argmax}")
    print(f"Using featured input: {need_featured_input}")

    # --- Setup directories and files ---
    base_dir = Path("resources") / "benchmark" / title
    base_dir.mkdir(parents=True, exist_ok=True)
    csv_path = base_dir / "metrics.csv"
    log_path = base_dir / "log.txt"

    print(f"Results will be saved to: {base_dir}")

    # --- Move model to device and set to eval mode ---
    print("Moving model to device and setting to eval mode...")
    model.to(device)
    model.eval()

    # --- Initialize accumulators ---
    all_preds = []
    all_labels = []
    total_time = 0.0

    # Reset peak memory stats before inference
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        print(f"CUDA memory stats reset. Using GPU: {torch.cuda.get_device_name(0)}")

    # 获取数据集大小用于进度显示
    try:
        dataset_size = len(dataloader.dataset)
        print(f"Dataset size: {dataset_size}")
    except:
        dataset_size = len(dataloader)  # 如果无法获取实际数据集大小，使用dataloader长度
        print(f"Dataset size: ~{dataset_size} (estimated)")

    # --- Run inference over entire dataset ---
    print("Starting inference...")

    # 使用tqdm显示进度
    if tqdm is not None:
        progress_bar = tqdm(enumerate(dataloader),
                            total=len(dataloader),
                            desc="Inference Progress",
                            unit="batch")
    else:
        progress_bar = enumerate(dataloader)
        print("Processing batches...")

    with torch.no_grad():
        for batch_idx, (feature_input, label, image) in progress_bar:
            # 更新进度条描述
            if tqdm is not None and batch_idx % 10 == 0:  # 每10个批次更新一次描述
                progress_bar.set_postfix({
                    'Batch': f'{batch_idx}',
                    'Time': f'{total_time:.2f}s'
                })

            if need_featured_input:
                feature_input = feature_input.to(device)
            else:
                feature_input = image.to(device)
            label = label.to(device)  # [B, 1, H, W]

            # Time each batch
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.time()

            output = model(feature_input)  # [B, n_classes, H, W] or [B, H, W] if no softmax

            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            batch_time = (end_time - start_time)
            total_time += batch_time

            # Post-process output to get predicted class indices [B, H, W]
            if need_argmax:
                if output.shape[1] == n_classes:
                    pred = torch.argmax(output, dim=1)  # [B, H, W]
                else:
                    raise ValueError(f"Expected output channel dimension to be {n_classes} when need_argmax=True.")
            else:
                # Assume output is already class indices (e.g., from a non-probabilistic model)
                pred = output.squeeze(1)  # [B, H, W]

            # Ensure label is [B, H, W] (remove channel dim)
            label = label.squeeze(1)  # [B, H, W]

            # Move to CPU and flatten
            pred_np = pred.cpu().numpy().flatten()
            label_np = label.cpu().numpy().flatten()

            all_preds.append(pred_np)
            all_labels.append(label_np)

            # 每100个批次打印一次状态
            if batch_idx % 100 == 0:
                print(f"Processed batch {batch_idx}, current total time: {total_time:.2f}s, "
                      f"time per batch: {batch_time:.4f}s")

    print(f"Inference completed! Total time: {total_time:.2f}s")

    # --- Concatenate all predictions and labels ---
    print("Concatenating predictions and labels...")
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    print(f"Final arrays shape - Predictions: {all_preds.shape}, Labels: {all_labels.shape}")

    # --- Compute metrics ---
    # Ignore invalid labels (e.g., if label has values >= n_classes or < 0)
    print("Computing metrics...")
    valid_mask = (all_labels >= 0) & (all_labels < n_classes)
    all_preds = all_preds[valid_mask]
    all_labels = all_labels[valid_mask]
    print(f"Valid pixels after filtering: {len(all_labels)}")

    # Pixel Accuracy
    pixel_acc = np.mean(all_preds == all_labels)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(n_classes)))

    # Class-wise IoU
    iou_per_class = []
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        denom = tp + fp + fn
        iou = tp / denom if denom != 0 else 0.0
        iou_per_class.append(iou)
    mIoU = np.mean(iou_per_class)

    # Class-wise Accuracy (diagonal of normalized CM)
    class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)

    # Classification report for precision, recall, f1
    report_dict = classification_report(
        all_labels, all_preds,
        labels=list(range(n_classes)),
        zero_division=0,
        output_dict=True
    )

    # Extract macro averages
    macro_precision = report_dict['macro avg']['precision']
    macro_recall = report_dict['macro avg']['recall']
    macro_f1 = report_dict['macro avg']['f1-score']

    # --- Timing & Hardware Info ---
    num_samples = len(dataloader.dataset)
    time_per_frame = total_time / num_samples
    fps = 1.0 / time_per_frame if time_per_frame > 0 else float('inf')

    if device.type == 'cuda':
        max_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        gpu_name = torch.cuda.get_device_name(0)
    else:
        max_memory_mb = 0.0
        gpu_name = "N/A"

    cpu_info = platform.processor() or "Unknown CPU"
    torch_version = torch.__version__

    # --- Prepare result dictionary ---
    result = {
        "model_title": title,
        "n_classes": n_classes,
        "num_samples": num_samples,
        "pixel_accuracy": pixel_acc,
        "mIoU": mIoU,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1_score": macro_f1,
        "total_inference_time_sec": total_time,
        "time_per_frame_sec": time_per_frame,
        "fps": fps,
        "max_gpu_memory_mb": max_memory_mb,
        "hardware_cpu": cpu_info,
        "hardware_gpu": gpu_name,
        "pytorch_version": torch_version,
    }

    # Add per-class metrics
    for i in range(n_classes):
        result[f"class_{i}_iou"] = iou_per_class[i]
        result[f"class_{i}_accuracy"] = class_acc[i]
        result[f"class_{i}_precision"] = report_dict.get(str(i), {}).get('precision', 0.0)
        result[f"class_{i}_recall"] = report_dict.get(str(i), {}).get('recall', 0.0)
        result[f"class_{i}_f1"] = report_dict.get(str(i), {}).get('f1-score', 0.0)

    # --- Write to CSV ---
    print("Writing results to CSV...")
    file_exists = csv_path.exists()
    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)

    # --- Format log message ---
    log_lines = []
    log_lines.append("=" * 60)
    log_lines.append(f"Benchmark Results: {title}")
    log_lines.append("=" * 60)
    log_lines.append(f"Pixel Accuracy: {pixel_acc:.4f}")
    log_lines.append(f"mIoU:           {mIoU:.4f}")
    log_lines.append(f"Macro F1:       {macro_f1:.4f}")
    log_lines.append(f"Total Time:     {total_time:.2f}s")
    log_lines.append(f"Time/Frame:     {time_per_frame * 1000:.2f} ms")
    log_lines.append(f"FPS:            {fps:.2f}")
    log_lines.append(f"Max GPU Mem:    {max_memory_mb:.1f} MB")
    log_lines.append("-" * 60)
    log_lines.append("Per-class IoU:")
    for i, iou in enumerate(iou_per_class):
        log_lines.append(f"  Class {i}: {iou:.4f}")
    log_lines.append("=" * 60)

    log_message = "\n".join(log_lines)

    # --- Print to console ---
    print(log_message)

    # --- Write to log.txt ---
    print("Writing results to log file...")
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(log_message + "\n")

    print(f"\n✅ Results saved to:\n   CSV: {csv_path}\n   LOG: {log_path}")
    print(f"✅ Benchmark completed successfully!")

    # --- 清理内存 ---
    print("Cleaning up memory...")
    del all_preds, all_labels, pred_np, label_np, output, pred, label, feature_input
    if device.type == 'cuda':
        torch.cuda.empty_cache()  # 清空CUDA缓存
    print("Memory cleanup completed.")


def evaluate_model_on_dataset(
        title: str,
        model: torch.nn.Module,
        dataset: torch.utils.data.Dataset,
        need_argmax: bool,
        device: torch.device,
        n_classes: int = 5,
        num_workers: int = 0,
        pin_memory: bool = False,
        need_featured_input: bool = True,
):
    """
    Evaluate a segmentation model on a given dataset using the benchmark function.

    Args:
        title (str): Experiment name; used for result folder naming.
        model (nn.Module): The model to evaluate.
        dataset (Dataset): A dataset that returns (feature_input, label, image).
                           - feature_input: [22, H, W]
                           - label: [1, H, W] with integer class indices in [0, n_classes-1]
                           - image: [3, H, W] (ignored)
        need_argmax (bool): Whether to apply argmax on model output.
        device (torch.device): Device to run inference on (e.g., torch.device('cuda')).
        n_classes (int): Number of semantic classes. Default: 5.
        num_workers (int): Number of subprocesses for data loading. Default: 0.
        pin_memory (bool): If True, enables faster GPU transfer. Default: False.

    Returns:
        None. Results are saved to disk.
    """
    print(f"Setting up DataLoader for dataset with {len(dataset)} samples...")

    # Create DataLoader with batch_size=1 as required
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,  # No need to shuffle during evaluation
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False  # Keep all samples
    )

    print(f"DataLoader created with {len(dataloader)} batches")

    # Call the benchmark function
    benchmark(
        title=title,
        model=model,
        dataloader=dataloader,
        need_argmax=need_argmax,
        device=device,
        n_classes=n_classes,
        need_featured_input=need_featured_input,
    )

    # --- 在evaluate_model_on_dataset函数结束后也清理内存 ---
    print("Cleaning up memory after evaluation...")
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    print("Memory cleanup completed after evaluation.")