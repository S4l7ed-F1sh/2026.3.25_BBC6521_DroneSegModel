import torch
from tqdm import tqdm
import numpy as np
from src.dataset.CheckDataset import check_dataset
from src.dataset.out_conv_dataset.OutConvDs import OutConvDs
from torchvision import transforms


def validate_channel_meaning(
        dataset,
        device,
        num_classes=5,
        num_batches_to_test=20  # 只需测试少量batch即可得出结论
):
    """
    验证多分支模型输出中，每个类别两个通道的含义（哪个代表“是”，哪个代表“否”）。

    Args:
        dataset: PyTorch Dataset 对象
        device: torch.device
        num_classes: 类别总数
        num_batches_to_test: 测试多少个batch
    """
    print(f"开始验证通道含义，测试 {num_batches_to_test} 个批次...")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # 存储每个类别的两种策略得分
    results = {}
    for cls in range(num_classes):
        results[cls] = {
            'argmax_0_accs': [],
            'argmax_0_mious': [],
            'argmax_1_accs': [],
            'argmax_1_mious': [],
        }

    batch_count = 0
    for feature_map, label, raw_image in dataloader:
        if batch_count >= num_batches_to_test:
            break

        feature_map = feature_map.to(device)
        target = label.squeeze(0).squeeze(0).long().to(device)
        probs = feature_map.squeeze(0)
        h, w = probs.shape[1], probs.shape[2]

        for cls in range(num_classes):
            start_idx = cls * 2
            cls_probs = probs[start_idx:start_idx + 2, :, :]  # Shape: [2, H, W]

            # --- 测试 argmax == 0 代表“是” ---
            pred_argmax_0_is_positive = (torch.argmax(cls_probs, dim=0) == 0)
            mask_argmax_0 = (target == cls)
            metrics_0 = calculate_binary_metrics(pred_argmax_0_is_positive, mask_argmax_0)
            results[cls]['argmax_0_accs'].append(metrics_0['acc'])
            results[cls]['argmax_0_mious'].append(metrics_0['iou'])

            # --- 测试 argmax == 1 代表“是” ---
            pred_argmax_1_is_positive = (torch.argmax(cls_probs, dim=0) == 1)
            mask_argmax_1 = (target == cls)
            metrics_1 = calculate_binary_metrics(pred_argmax_1_is_positive, mask_argmax_1)
            results[cls]['argmax_1_accs'].append(metrics_1['acc'])
            results[cls]['argmax_1_mious'].append(metrics_1['iou'])

        batch_count += 1
        if batch_count % max(1, num_batches_to_test // 10) == 0:
            print(f"  已测试 {batch_count}/{num_batches_to_test} 个批次...")

    print("\n" + "=" * 80)
    print(
        f"{'类别':<4} | {'argmax==0 (Acc)':<15} | {'argmax==0 (mIoU)':<15} | {'argmax==1 (Acc)':<15} | {'argmax==1 (mIoU)':<15} | {'推荐策略'}")
    print("-" * 80)
    for cls in range(num_classes):
        mean_acc_0 = np.mean(results[cls]['argmax_0_accs'])
        mean_miou_0 = np.mean(results[cls]['argmax_0_mious'])
        mean_acc_1 = np.mean(results[cls]['argmax_1_accs'])
        mean_miou_1 = np.mean(results[cls]['argmax_1_mious'])

        recommended_strategy = "argmax==1 (是)" if mean_miou_1 > mean_miou_0 else "argmax==0 (是)"

        print(
            f"{cls:<4} | {mean_acc_0:<15.4f} | {mean_miou_0:<15.4f} | {mean_acc_1:<15.4f} | {mean_miou_1:<15.4f} | {recommended_strategy}")

    print("=" * 80)


def calculate_binary_metrics(pred_mask, true_mask):
    """
    计算二分类掩膜的 Accuracy 和 IoU。
    pred_mask: bool tensor [H, W]
    true_mask: bool tensor [H, W] (对应类别的 ground truth)
    """
    pred_flat = pred_mask.view(-1).long()
    true_flat = true_mask.view(-1).long()

    tp = (pred_flat & true_flat).sum().item()
    fp = (pred_flat & (~true_flat)).sum().item()
    tn = ((~pred_flat) & (~true_flat)).sum().item()
    fn = ((~pred_flat) & true_flat).sum().item()

    total_pixels = true_flat.numel()
    acc = (tp + tn) / total_pixels if total_pixels > 0 else 0.0

    union = tp + fp + fn
    iou = tp / union if union > 0 else 1.0  # 如果预测和真实都是空集，IoU定义为1

    return {'acc': acc, 'iou': iou}

import os
def run_validation():
    """
    主运行函数
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用设备: {device}")


    pj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    para_dir = os.path.join(pj_root, 'resources', 'para')

    para_list = [
        os.path.join(para_dir, 'unet_branch0.pth'),
        os.path.join(para_dir, 'unet_branch1.pth'),
        os.path.join(para_dir, 'unet_branch2.pth'),
        os.path.join(para_dir, 'unet_branch3.pth'),
        os.path.join(para_dir, 'unet_branch3.pth'),
    ]

    # --- 加载你的数据集 ---
    # 注意：这里需要与你原代码中的数据集加载方式保持一致
    ds_path = check_dataset()
    dataset = OutConvDs(
        image_dir='drone_seg_dataset/classes_dataset/classes_dataset/original_images',
        label_dir='drone_seg_dataset/classes_dataset/classes_dataset/label_images_semantic',
        # 注意：这个验证函数不需要模型参数列表
        model_param_list=para_list,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
    )

    validate_channel_meaning(dataset, device, num_classes=5, num_batches_to_test=20)


if __name__ == "__main__":
    run_validation()