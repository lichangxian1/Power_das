"""
特征增强脚本：为现有数据集添加 arrival_time + input_skew 特征
将 X: [N, 7] 扩展为 X: [N, 9]

新增特征：
  - arrival_time:  节点输出稳定时间（静态时序模型）
  - input_skew:    节点输入到达时间的最大差异（毛刺直接指示器）

不需要重跑 EDA，直接基于已有 (X, P) 在原数据集上扩展。
"""

import os
import torch
from tqdm import tqdm


# 节点类型 → 单元延迟（unit-less）
# type 0 = 3:2 compressor (Full Adder)，路径 ≈ 2 个门
# type 1 = 2:2 compressor (Half Adder)，路径 ≈ 1 个门
# type 2 = PP (Partial Product)，作为输入，延迟为 0
# type 3 = output / 其它
DEFAULT_TYPE_DELAYS = (2.0, 1.0, 0.0, 0.5)


def compute_timing_features(X, P, type_delays=DEFAULT_TYPE_DELAYS):
    """
    Args:
        X: [N, 7]  原始节点特征 (stage_idx, col_idx, idx, type_onehot×4)
        P: [N, N]  邻接矩阵 P[src, dst]=1 表示 src→dst
    Returns:
        arrival: [N]  每个节点输出稳定的时间
        skew:    [N]  每个节点输入到达时间的最大差（max-min）
    """
    N = X.shape[0]
    type_delays_t = torch.tensor(type_delays, dtype=torch.float32)
    node_types = X[:, 3:7].argmax(dim=1)              # [N]
    own_delays = type_delays_t[node_types]            # [N]

    # 拓扑序：把 PP 强制排到最前，其余按 stage_idx 升序
    sort_key = X[:, 0].clone().float()
    sort_key[node_types == 2] = -1e9
    order = torch.argsort(sort_key, stable=True)

    arrival = torch.zeros(N, dtype=torch.float32)
    for v_t in order:
        v = v_t.item()
        if node_types[v] == 2:  # PP 视为 0
            arrival[v] = 0.0
            continue
        preds_mask = P[:, v] > 0.5
        if preds_mask.any():
            arrival[v] = arrival[preds_mask].max() + own_delays[v]
        else:
            arrival[v] = own_delays[v]

    # 输入到达时间偏差（毛刺指示器）
    skew = torch.zeros(N, dtype=torch.float32)
    for v in range(N):
        preds_mask = P[:, v] > 0.5
        if int(preds_mask.sum().item()) >= 2:
            pa = arrival[preds_mask]
            skew[v] = (pa.max() - pa.min()).item()

    return arrival, skew


def augment_dataset(in_path, out_path, type_delays=DEFAULT_TYPE_DELAYS, force=False):
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"找不到输入文件: {in_path}")

    data = torch.load(in_path, map_location="cpu")
    print(f"  📂 加载 {len(data)} 个样本: {in_path}")

    # 已增强过的保护
    feat_dim = data[0]["X"].shape[1]
    if feat_dim >= 9 and not force:
        print(f"  ⚠️ 检测到 X 已有 {feat_dim} 维特征，似乎已经增强过")
        print(f"     如需强制覆盖请传 force=True，或先备份")
        return

    all_arrivals, all_skews = [], []
    fail_count = 0

    for item in tqdm(data, desc="计算 timing 特征"):
        try:
            X = item["X"]
            P = item["P"]

            arrival, skew = compute_timing_features(X, P, type_delays)

            # 始终从原始 7 维基础上扩展（防止反复增强累加）
            X_base = X[:, :7]
            X_new = torch.cat(
                [X_base, arrival.unsqueeze(1), skew.unsqueeze(1)], dim=1
            )
            item["X"] = X_new

            all_arrivals.append(arrival)
            all_skews.append(skew)
        except Exception as e:
            fail_count += 1
            if fail_count <= 5:
                print(f"  ⚠️ 样本处理失败: {e}")

    # 统计
    all_arrival = torch.cat(all_arrivals)
    all_skew = torch.cat(all_skews)
    print()
    print(f"  📊 Timing 特征分布:")
    print(f"     arrival_time:  mean={all_arrival.mean():.3f}  "
          f"max={all_arrival.max():.3f}  std={all_arrival.std():.3f}")
    print(f"     input_skew:    mean={all_skew.mean():.3f}  "
          f"max={all_skew.max():.3f}  std={all_skew.std():.3f}")
    nonzero_pct = (all_skew > 0).float().mean() * 100
    print(f"     skew > 0 节点占比: {nonzero_pct:.1f}%  (这部分节点是潜在毛刺源)")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(data, out_path)
    print()
    print(f"  💾 已保存: {out_path}")
    print(f"     新 X 维度: {list(data[0]['X'].shape)}")
    print(f"     失败样本: {fail_count}")


if __name__ == "__main__":
    augment_dataset(
        in_path="dataset/glitch_power_data_16bit.pt",
        out_path="dataset/glitch_power_data_16bit_v2.pt",
    )
