from tqdm import tqdm
from itertools import permutations
from src.dataset.out_conv_dataset.OutConvDs import OutConvDs
import os
import torch
from src.dataset.CheckDataset import check_dataset
from torchvision import transforms
import numpy as np
import json
from src.training.out_conv_training.SortingMethod import get_mask_from_permutation


def calculate_metrics(pred, target, num_classes=5):
    """
    计算语义分割的各项指标：Accuracy, mIoU, Precision, Recall, F1-score。
    """
    if pred.is_cuda:
        pred = pred.cpu()
    if target.is_cuda:
        target = target.cpu()

    pred = pred.long()
    target = target.long()

    acc = (pred == target).sum().item() / target.numel()
    ious = []
    precisions = []
    recalls = []
    f1_scores = []

    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()
        pred_sum = pred_mask.sum().item()
        target_sum = target_mask.sum().item()

        if union == 0:
            iou = 1.0 if pred_sum == 0 and target_sum == 0 else 0.0
        else:
            iou = intersection / union
        ious.append(iou)

        if pred_sum == 0:
            precision = 1.0 if target_sum == 0 else 0.0
        else:
            precision = intersection / pred_sum

        if target_sum == 0:
            recall = 1.0 if pred_sum == 0 else 0.0
        else:
            recall = intersection / target_sum

        precisions.append(precision)
        recalls.append(recall)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        f1_scores.append(f1)

    return {
        'acc': acc,
        'miou': np.mean(ious),
        'precision': np.mean(precisions),
        'recall': np.mean(recalls),
        'f1': np.mean(f1_scores)
    }


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用设备: {device}")

    pj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    para_dir = os.path.join(pj_root, 'resources', 'para')

    para_list = [
        os.path.join(para_dir, 'unet_branch0.pth'),
        os.path.join(para_dir, 'unet_branch1.pth'),
        os.path.join(para_dir, 'unet_branch2.pth'),
        os.path.join(para_dir, 'unet_branch3.pth'),
        os.path.join(para_dir, 'unet_branch4.pth'),
    ]

    ds_path = check_dataset()
    dataset = OutConvDs(
        image_dir='drone_seg_dataset/classes_dataset/classes_dataset/original_images',
        label_dir='drone_seg_dataset/classes_dataset/classes_dataset/label_images_semantic',
        model_param_list=para_list,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    class_permutations = list(permutations(range(5)))
    print(f"共有 {len(class_permutations)} 种优先级排列，开始评估...")

    # --- 文件路径准备 ---
    output_dir = os.path.join(pj_root, 'resources', 'results')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    json_path = os.path.join(output_dir, 'permutation_results.json')

    # 如果文件已存在，先备份或删除，防止追加写入导致数据混乱
    if os.path.exists(json_path):
        os.remove(json_path)
        print(f"已清理旧文件: {json_path}")

    all_results = []  # 存储所有结果以便后续分析

    # --- 外层循环 ---
    for i, perm in enumerate(tqdm(class_permutations, desc="Testing Permutations")):
        total_acc = 0.0
        total_miou = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        count = 0

        # --- 内层循环 ---
        for feature_map, label, raw_image in dataloader:
            feature_map = feature_map.to(device)
            target = label.squeeze(0).squeeze(0).long().to(device)

            # 调用重构的函数生成掩膜
            final_pred = get_mask_from_permutation(feature_map, perm)

            metrics = calculate_metrics(final_pred, target, num_classes=5)
            total_acc += metrics['acc']
            total_miou += metrics['miou']
            total_precision += metrics['precision']
            total_recall += metrics['recall']
            total_f1 += metrics['f1']
            count += 1

        # --- 计算当前排列的平均指标 ---
        avg_metrics = {
            'perm': perm,
            'acc': total_acc / count,
            'miou': total_miou / count,
            'precision': total_precision / count,
            'recall': total_recall / count,
            'f1': total_f1 / count,
            'sum_three': (total_miou / count) + (total_f1 / count) + (total_acc / count)  # 三者之和
        }

        all_results.append(avg_metrics)

        # 实时打印当前最佳（可选，这里简单打印一下 mIoU）
        tqdm.write(f"完成排列 {perm} -> mIoU: {avg_metrics['miou']:.4f}")

    # 保存所有结果到JSON文件
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    print(f"\n✅ 所有计算完成，详细结果已保存至: {json_path}")

    # --- 分别按不同指标排序并找出最佳排列 ---

    # 1. 按mIoU排序（最高在前）
    results_by_miou = sorted(all_results, key=lambda x: x['miou'], reverse=True)
    best_miou_result = results_by_miou[0]

    # 2. 按Pixel Acc排序（最高在前）
    results_by_acc = sorted(all_results, key=lambda x: x['acc'], reverse=True)
    best_acc_result = results_by_acc[0]

    # 3. 按F1-score排序（最高在前）
    results_by_f1 = sorted(all_results, key=lambda x: x['f1'], reverse=True)
    best_f1_result = results_by_f1[0]

    # 4. 按三者之和排序（最高在前）
    results_by_sum = sorted(all_results, key=lambda x: x['sum_three'], reverse=True)
    best_sum_result = results_by_sum[0]

    # --- 保存各种最佳结果 ---
    # 创建汇总结果字典
    summary_results = {
        'best_miou': {
            'perm': best_miou_result['perm'],
            'miou': best_miou_result['miou'],
            'acc': best_miou_result['acc'],
            'f1': best_miou_result['f1'],
            'sum_three': best_miou_result['sum_three']
        },
        'best_acc': {
            'perm': best_acc_result['perm'],
            'miou': best_acc_result['miou'],
            'acc': best_acc_result['acc'],
            'f1': best_acc_result['f1'],
            'sum_three': best_acc_result['sum_three']
        },
        'best_f1': {
            'perm': best_f1_result['perm'],
            'miou': best_f1_result['miou'],
            'acc': best_f1_result['acc'],
            'f1': best_f1_result['f1'],
            'sum_three': best_f1_result['sum_three']
        },
        'best_sum': {
            'perm': best_sum_result['perm'],
            'miou': best_sum_result['miou'],
            'acc': best_sum_result['acc'],
            'f1': best_sum_result['f1'],
            'sum_three': best_sum_result['sum_three']
        }
    }

    # 保存汇总结果
    summary_json_path = os.path.join(output_dir, 'best_results_summary.json')
    with open(summary_json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 100)
    print(f"{'排序方式':<12} | {'排列':<20} | {'mIoU':<8} | {'Acc':<8} | {'F1':<8} | {'三者之和':<10}")
    print("-" * 100)
    print(
        f"{'mIoU最优':<12} | {str(best_miou_result['perm']):<20} | {best_miou_result['miou']:<8.4f} | {best_miou_result['acc']:<8.4f} | {best_miou_result['f1']:<8.4f} | {best_miou_result['sum_three']:<10.4f}")
    print(
        f"{'Pixel Acc最优':<12} | {str(best_acc_result['perm']):<20} | {best_acc_result['miou']:<8.4f} | {best_acc_result['acc']:<8.4f} | {best_acc_result['f1']:<8.4f} | {best_acc_result['sum_three']:<10.4f}")
    print(
        f"{'F1-score最优':<12} | {str(best_f1_result['perm']):<20} | {best_f1_result['miou']:<8.4f} | {best_f1_result['acc']:<8.4f} | {best_f1_result['f1']:<8.4f} | {best_f1_result['sum_three']:<10.4f}")
    print(
        f"{'三者之和最优':<12} | {str(best_sum_result['perm']):<20} | {best_sum_result['miou']:<8.4f} | {best_sum_result['acc']:<8.4f} | {best_sum_result['f1']:<8.4f} | {best_sum_result['sum_three']:<10.4f}")
    print("=" * 100)

    # --- 示例：如何在找到最佳排列后使用新函数 ---
    print("\n--- 示例：使用各指标对应的最优排列生成单张图片的预测掩膜 ---")
    # 假设我们想对数据集中的第一张图片应用各个最优排列
    sample_feature_map, sample_label, _ = next(iter(dataloader))
    sample_feature_map = sample_feature_map.to(device)

    # 使用mIoU最优排列
    best_miou_perm = tuple(best_miou_result['perm'])
    predicted_mask_miou = get_mask_from_permutation(sample_feature_map, best_miou_perm)
    print(
        f"使用mIoU最优排列 {best_miou_perm}，对第一张图片生成的预测掩膜形状: {predicted_mask_miou.shape}, 设备: {predicted_mask_miou.device}")
    print(f"预测掩膜的唯一值 (类别): {torch.unique(predicted_mask_miou)}")

    # 使用Pixel Acc最优排列
    best_acc_perm = tuple(best_acc_result['perm'])
    predicted_mask_acc = get_mask_from_permutation(sample_feature_map, best_acc_perm)
    print(
        f"使用Pixel Acc最优排列 {best_acc_perm}，对第一张图片生成的预测掩膜形状: {predicted_mask_acc.shape}, 设备: {predicted_mask_acc.device}")
    print(f"预测掩膜的唯一值 (类别): {torch.unique(predicted_mask_acc)}")

    # 使用F1-score最优排列
    best_f1_perm = tuple(best_f1_result['perm'])
    predicted_mask_f1 = get_mask_from_permutation(sample_feature_map, best_f1_perm)
    print(
        f"使用F1-score最优排列 {best_f1_perm}，对第一张图片生成的预测掩膜形状: {predicted_mask_f1.shape}, 设备: {predicted_mask_f1.device}")
    print(f"预测掩膜的唯一值 (类别): {torch.unique(predicted_mask_f1)}")

    # 使用三者之和最优排列
    best_sum_perm = tuple(best_sum_result['perm'])
    predicted_mask_sum = get_mask_from_permutation(sample_feature_map, best_sum_perm)
    print(
        f"使用三者之和最优排列 {best_sum_perm}，对第一张图片生成的预测掩膜形状: {predicted_mask_sum.shape}, 设备: {predicted_mask_sum.device}")
    print(f"预测掩膜的唯一值 (类别): {torch.unique(predicted_mask_sum)}")


if __name__ == "__main__":
    test()