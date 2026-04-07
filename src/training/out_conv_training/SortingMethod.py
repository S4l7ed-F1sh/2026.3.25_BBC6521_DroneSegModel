import torch

BEST_PERM = [1, 2, 0, 3, 4]

def get_mask_from_permutation(feature_map: torch.Tensor, class_permutation: tuple):
    """
    根据给定的类别优先级排列，从特征图生成最终的预测掩膜。

    Args:
        feature_map (torch.Tensor): 形状为 [2*N, H, W] 的张量，其中 N 是类别数。
                                    每两个通道对应一个类别的 [背景, 前景] 概率。
        class_permutation (tuple): 包含类别索引的排列元组，表示优先级顺序。
                                   例如 (2, 0, 3, 1, 4) 表示类别 2 优先级最高，类别 4 最低。

    Returns:
        torch.LongTensor: 形状为 [H, W] 的预测掩膜张量。
    """
    device = feature_map.device
    _, h, w = feature_map.shape

    # 将 N 个二分类分支的结果分离
    # probs.shape = [N, 2, H, W]
    probs = feature_map.view(len(class_permutation), 2, h, w)

    # 初始化最终预测图，默认为优先级最低的类别
    final_pred = torch.full((h, w), fill_value=class_permutation[-1], dtype=torch.long, device=device)

    # 按优先级从高到低（排列中的第一个到最后一个）处理
    for priority_idx, cls in enumerate(class_permutation[:-1]):  # 排除最后一个
        # 获取当前类别的二分类概率 [背景, 前景]
        cls_probs = probs[cls, :, :, :]  # shape [2, H, W]

        # 判定为前景的位置 (argmax == 1)
        is_positive = (torch.argmax(cls_probs, dim=0) == 1)  # shape [H, W]

        # 在这些位置上更新最终预测
        final_pred[is_positive] = cls

    return final_pred
