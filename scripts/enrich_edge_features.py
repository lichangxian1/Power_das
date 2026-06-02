"""
方案 1: 边特征扩展 (edge_attr [E,5] → [E,10])

新增 5 维, 用 X[N,13] 和 edge_index 计算:
  5: edge_skew      = (arrival[src] - arrival[dst]) / NORM
  6: src_arrival    = arrival[src] / NORM
  7: dst_arrival    = arrival[dst] / NORM
  8: src_fanout     = fanout[src] / MAX_FANOUT
  9: dst_fanin       = fanin[dst] / 3.0   (FA 最大 3 入度, 区分 FA/HA/末级)

X 维度索引 (来自 enrich_node_power_dataset):
  X[:, 0] = stage_idx
  X[:, 1] = col_idx
  X[:, 2] = idx
  X[:, 3:7] = type_onehot
  X[:, 7] = arrival_time
  X[:, 8] = input_skew
  X[:, 9] = fanout
  X[:, 10] = fanin
  X[:, 11] = depth
  X[:, 12] = is_critical

用法:
    python scripts/enrich_edge_features.py \
        --in_path dataset/glitch_power_data_16bit_v2_7k_enriched.pt \
        --out_path dataset/glitch_power_data_16bit_v2_7k_edge10.pt
"""
import os
import argparse
import torch
from tqdm import tqdm


# 归一化常数 (跨样本统一, 避免 fold 间漂移)
ARRIVAL_NORM = 30.0       # arrival_time 经验范围 [0, 30]
FANOUT_NORM = 16.0        # PP 节点出度 max ~16; FA/HA 通常 1-2


def add_edge_features(X, edge_index, edge_attr_old):
    """
    Args:
        X:              [N, 13]  enriched 节点特征
        edge_index:     [2, E]
        edge_attr_old:  [E, 5]   原 [is_sum, is_carry, port_a, port_b, port_c]
    Returns:
        edge_attr_new:  [E, 10]  原 5 维 + 新 5 维
    """
    src = edge_index[0].long()
    dst = edge_index[1].long()

    arrival = X[:, 7]
    fanout = X[:, 9]
    fanin = X[:, 10]

    edge_skew = (arrival[src] - arrival[dst]) / ARRIVAL_NORM
    src_arrival = arrival[src] / ARRIVAL_NORM
    dst_arrival = arrival[dst] / ARRIVAL_NORM
    src_fanout = fanout[src] / FANOUT_NORM
    dst_fanin = fanin[dst] / 3.0   # FA 最大入度 3

    new_features = torch.stack([
        edge_skew, src_arrival, dst_arrival, src_fanout, dst_fanin
    ], dim=1)  # [E, 5]

    return torch.cat([edge_attr_old, new_features], dim=1)


def main(in_path, out_path):
    data = torch.load(in_path, map_location="cpu", weights_only=False)
    print(f"  📂 加载 {len(data)} 样本: {in_path}")
    sample = data[0]
    print(f"     X 维度:          {list(sample['X'].shape)}")
    print(f"     旧 edge_attr:    {list(sample['edge_attr'].shape)}")

    if sample["X"].shape[1] < 13:
        raise ValueError(f"X 维度 {sample['X'].shape[1]} < 13, 请先跑 enrich_node_power_dataset")
    if sample["edge_attr"].shape[1] != 5:
        print(f"  ⚠️ edge_attr 已经是 {sample['edge_attr'].shape[1]} 维, 可能已扩展")

    for item in tqdm(data, desc="扩展 edge_attr 5→10"):
        item["edge_attr"] = add_edge_features(
            item["X"], item["edge_index"], item["edge_attr"]
        )

    print(f"\n  ✅ 扩展完成")
    print(f"     新 edge_attr:    {list(data[0]['edge_attr'].shape)}")
    print(f"  💾 保存到: {out_path}")
    torch.save(data, out_path)

    # 统计新特征的分布
    print(f"\n  📊 新增 5 维统计 (从 sample 0):")
    ea = data[0]["edge_attr"]
    names = ["edge_skew", "src_arrival", "dst_arrival", "src_fanout", "dst_fanin"]
    for i, name in enumerate(names):
        v = ea[:, 5 + i]
        print(f"     [{5+i}] {name:18s}: "
              f"min={v.min():+.3f}, max={v.max():+.3f}, "
              f"mean={v.mean():+.3f}, std={v.std():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path",
                        default="dataset/glitch_power_data_16bit_v2_7k_enriched.pt")
    parser.add_argument("--out_path",
                        default="dataset/glitch_power_data_16bit_v2_7k_edge10.pt")
    args = parser.parse_args()
    main(args.in_path, args.out_path)
