import math


def format_duration(seconds):
    """
    将秒数转换为 'a h, b min, c s' 格式的字符串。
    规则：
    1. 秒数保留2位小数（直接截断，不四舍五入）。
    2. 小时(a)和分钟(b)如果是0则省略。
    3. 秒(c)始终显示，且可以是浮点数。
    """

    # 1. 截断保留2位小数
    # 使用 math.floor(seconds * 100) / 100.0 实现直接截断（向下取整）
    # 例如 3661.999 -> 3661.99, 5.999 -> 5.99
    truncated_seconds = math.floor(seconds * 100) / 100.0

    # 2. 计算时、分、秒
    hours = int(truncated_seconds // 3600)
    minutes = int((truncated_seconds % 3600) // 60)
    secs = truncated_seconds % 60

    # 3. 构建字符串部分
    parts = []

    # 添加小时
    if hours > 0:
        parts.append(f"{hours} h")

    # 添加分钟
    if minutes > 0:
        parts.append(f"{minutes} min")

    # 添加秒 (始终添加)
    # 使用 :.2f 确保秒数即使被截断后也是两位小数显示 (如 5.1 -> 5.10)
    # 如果你希望 5.10 显示为 5.1，可以将 :.2f 改为 :g 或者手动处理，
    # 但通常保留 :.2f 最符合“保留两位小数”的视觉预期。
    parts.append(f"{secs:.2f} s")

    # 4. 拼接结果
    return ", ".join(parts)


# --- 测试代码 ---
if __name__ == "__main__":
    test_cases = [
        1.5,  # 只有秒
        65.12345,  # 有分，有秒（测试截断）
        3661.999,  # 有时，有分，有秒（测试截断进位）
        7200.05,  # 只有时和秒
        59.9,  # 临界值
        3599.99  # 临界值
    ]

    for t in test_cases:
        print(f"原始: {t} -> 格式化: {format_duration(t)}")