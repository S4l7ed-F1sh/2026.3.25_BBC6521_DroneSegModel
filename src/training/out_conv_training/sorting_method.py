from tqdm import tqdm
from itertools import permutations
from src.dataset.out_conv_dataset.OutConvDs import OutConvDs
import os
import torch
from src.dataset.CheckDataset import check_dataset
from torchvision import transforms
import numpy as np

def calculate_metrics(pred, target, num_classes=5):
    """
    计算单个样本的评估指标
    pred: [H, W] 预测图
    target: [H, W] 标签图
    """
    # 确保数据类型一致
    pred = pred.long()
    target = target.long()

    # 1. Pixel Accuracy
    acc = (pred == target).sum().item() / target.numel()

    # 2. mIoU, Precision, Recall, F1
    ious = []
    precisions = []
    recalls = []
    f1_scores = []

    for cls in range(num_classes):
        # 当前类别的预测和真值掩码
        pred_mask = (pred == cls)
        target_mask = (target == cls)

        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()
        pred_sum = pred_mask.sum().item()
        target_sum = target_mask.sum().item()

        # IoU
        if union == 0:
            iou = 1.0 if pred_sum == 0 and target_sum == 0 else 0.0  # 处理空类别情况
        else:
            iou = intersection / union
        ious.append(iou)

        # Precision & Recall
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

        # F1
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

def main():
    pj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    para_dir = os.path.join(pj_root, 'resources', 'para')

    para_list = [
        os.path.join(para_dir, 'unet_branch0.pth'),
        os.path.join(para_dir, 'unet_branch1.pth'),
        os.path.join(para_dir, 'unet_branch2.pth'),
        os.path.join(para_dir, 'unet_branch3.pth'),
        os.path.join(para_dir, 'unet_branch3.pth'),
    ]

    # 加载数据集
    ds_path = check_dataset()
    dataset = OutConvDs(
        image_dir='drone_seg_dataset/classes_dataset/classes_dataset/original_images',
        label_dir='drone_seg_dataset/classes_dataset/classes_dataset/label_images_semantic',
        model_param_list=para_list,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
        # data_enforcement=True,
    )

    # 创建 batchsize 为 1 的 DataLoader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # 遍历 DataLoader，对于每个样本，格式为 feature_map, label, image
    # feature_map 的形状为 [1, 10, H, W]，label 的形状为 [1, 1, H, W]，image 的形状为 [H, W, 3]
    # feature_map 是经过 5 个分支 U-Net 模型处理后的输出，每个模型的输出为 2 通道图，分别表示了该像素属于该类别的概率和不属于该类别的概率
    # 本代码的目的是验证最终多分支输出的一种融合方法：直接融合。具体来说，先将输出的 10 通道图拆为 5 个 2 通道图，然后进行 argmax，分别表示 是/否 属于第 i 类。
    # 然后，本代码将遍历 5 的所有排列，总共 120 种选择，分辨为每个类别分配一个优先级。每个像素的类别为优先级最高的，判定为是的类别。如果全部判定为否，则该像素类别为优先级最低的类别。
    # 最后，本代码将计算所有图像对这些图像进行评估，计算每个类别的 pixel accuracy和 mIoU，还有 准确率、召回率、F1 score 等指标。
    # 这个过程的目的是验证不同的类别优先级排序方法对最终评估指标的影响，从而找到一种最优的排序方法。

    # 生成 5 个类别的所有排列 (0, 1, 2, 3, 4)
    # 总共 120 种
    class_permutations = list(permutations(range(5)))

    print(f"共有 {len(class_permutations)} 种优先级排列，开始评估...")

    # 用于存储每种排列的总得分
    results = []

    # 外层循环：遍历每一种优先级排列
    # 为了演示效率，这里使用 tqdm 显示进度
    for perm in tqdm(class_permutations, desc="Testing Permutations"):
        total_acc = 0.0
        total_miou = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        count = 0

        # 内层循环：遍历数据集
        for feature_map, label, raw_image in dataloader:
            # feature_map shape: [1, 10, H, W]
            # label shape: [1, 1, H, W]

            # 1. 拆分通道
            # 将 [1, 10, H, W] 拆分为 5 个 [1, 2, H, W]
            # 假设通道顺序是: [class0_no, class0_yes, class1_no, class1_yes, ...]
            # 或者 [class0_yes, class0_no, ...] -> 请根据模型实际输出调整这里的切片
            # 这里假设每2个通道是一组，第1个是“是/前景”，第0个是“否/背景”的概率（或者反过来，需确认）
            # 通常 argmax 取最大值索引。如果索引0是“否”，索引1是“是”。

            probs = feature_map.squeeze(0)  # [10, H, W]
            h, w = probs.shape[1], probs.shape[2]

            # 最终预测图初始化
            final_pred = torch.full((h, w), -1, dtype=torch.long)

            # 2. 按照优先级 perm 进行融合
            # perm 例如: (2, 0, 4, 1, 3) -> 优先级最高是类别2，最低是类别3

            # 这里的逻辑：
            # 遍历优先级列表，如果当前类别判定为“是”，则赋予该类别，且不再被低优先级覆盖
            # 如果遍历完所有高优先级都是“否”，则赋予最低优先级的类别

            lowest_priority_class = perm[-1]

            # 初始化预测图为最低优先级类别（兜底策略）
            final_pred[:] = lowest_priority_class

            # 按照优先级从高到低遍历（除了最后一个，因为最后一个已经是默认值了）
            for cls in perm[:-1]:
                # 提取该类别的 2 个通道
                # 假设通道排列是交错的：0,1 是类别0; 2,3 是类别1...
                start_idx = cls * 2
                cls_probs = probs[start_idx:start_idx + 2, :, :]  # [2, H, W]

                # 进行 argmax，0表示“否”，1表示“是”
                # 取出“是”的通道索引是否为 1
                is_positive = (torch.argmax(cls_probs, dim=0) == 1)  # [H, W] 布尔值

                # 将判定为“是”的像素位置更新为当前类别
                final_pred[is_positive] = cls

            # 3. 计算指标
            # label 需要处理成 [H, W]
            target = label.squeeze(0).squeeze(0).long()

            metrics = calculate_metrics(final_pred, target, num_classes=5)

            total_acc += metrics['acc']
            total_miou += metrics['miou']
            total_precision += metrics['precision']
            total_recall += metrics['recall']
            total_f1 += metrics['f1']
            count += 1

        # 计算该排列的平均指标
        avg_metrics = {
            'perm': perm,
            'acc': total_acc / count,
            'miou': total_miou / count,
            'precision': total_precision / count,
            'recall': total_recall / count,
            'f1': total_f1 / count
        }
        results.append(avg_metrics)

    # 4. 结果排序与输出
    # 按 mIoU 降序排列
    results.sort(key=lambda x: x['miou'], reverse=True)

    print("\n" + "=" * 80)
    print(f"{'优先级排列':<20} | {'mIoU':<10} | {'Acc':<10} | {'F1':<10} | {'Prec':<10} | {'Rec':<10}")
    print("-" * 80)

    # 打印前 10 名
    for res in results[:10]:
        perm_str = str(res['perm'])
        print(
            f"{perm_str:<20} | {res['miou']:.4f}     | {res['acc']:.4f}     | {res['f1']:.4f}     | {res['precision']:.4f}     | {res['recall']:.4f}")

    print("=" * 80)
    print(f"最佳排列: {results[0]['perm']}，对应 mIoU: {results[0]['miou']:.4f}")


if __name__ == "__main__":
    main()