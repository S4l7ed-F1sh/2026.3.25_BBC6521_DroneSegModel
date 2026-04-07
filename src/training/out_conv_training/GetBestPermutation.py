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
        os.path.join(para_dir, 'unet_branch3.pth'),  # 注意：这里似乎重复了 branch3
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

    # 打开文件用于流式写入
    with open(json_path, 'w', encoding='utf-8') as f:
        f.write("[\n")  # 开始 JSON 数组
        first_item = True

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
                final_pred = get_mask_from_permutation(feature_map.squeeze(0), perm)

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
                'f1': total_f1 / count
            }

            # --- 实时写入文件 ---
            if not first_item:
                f.write(",\n")  # 添加逗号分隔符
            else:
                first_item = False

            # 写入当前结果
            json.dump(avg_metrics, f, indent=4, ensure_ascii=False)

            # 强制刷新缓冲区，确保数据写入硬盘
            f.flush()
            os.fsync(f.fileno())

            # 实时打印当前最佳（可选，这里简单打印一下 mIoU）
            tqdm.write(f"完成排列 {perm} -> mIoU: {avg_metrics['miou']:.4f}")

        f.write("\n]")  # 结束 JSON 数组
        f.flush()
        os.fsync(f.fileno())

    print(f"\n✅ 所有计算完成，详细结果已实时保存至: {json_path}")

    # --- 最后读取文件生成排行榜 ---
    # 因为我们是流式写入的，最后需要重新读取来排序
    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    results.sort(key=lambda x: x['miou'], reverse=True)
    best_res = results[0]

    # 保存最佳结果
    txt_path = os.path.join(output_dir, 'best_permutation.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"最佳排列: {best_res['perm']}\n")
        f.write(f"mIoU: {best_res['miou']:.4f}\n")

    print("\n" + "=" * 80)
    print(f"{'优先级排列':<20} | {'mIoU':<10} | {'Acc':<10}")
    print("-" * 80)
    for res in results[:5]:  # 只显示前5个
        print(f"{str(res['perm']):<20} | {res['miou']:.4f}     | {res['acc']:.4f}")
    print("=" * 80)
    print(f"🏆 最佳排列: {best_res['perm']}")

    # --- 示例：如何在找到最佳排列后使用新函数 ---
    print("\n--- 示例：使用最佳排列生成单张图片的预测掩膜 ---")
    # 假设我们想对数据集中的第一张图片应用最佳排列
    sample_feature_map, sample_label, _ = next(iter(dataloader))
    sample_feature_map = sample_feature_map.to(device)

    best_perm = tuple(best_res['perm'])  # 确保类型一致
    predicted_mask = get_mask_from_permutation(sample_feature_map.squeeze(0), best_perm)

    print(
        f"使用最佳排列 {best_perm}，对第一张图片生成的预测掩膜形状: {predicted_mask.shape}, 设备: {predicted_mask.device}")
    print(f"预测掩膜的唯一值 (类别): {torch.unique(predicted_mask)}")


if __name__ == "__main__":
    test()