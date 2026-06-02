"""合并 8-bit + 16-bit 数据 → 4-dim type onehot, 用于纯 GIN onehot 训练.

每个样本:
  X[N, 4]      — type onehot only (FA/HA/PP/output)
  edge_index   — 保留 (PureGIN 用 sum agg, 但 forward 也需要 edge_index 拓扑)
  edge_attr    — 保留 (PureGIN 忽略, 但保留以兼容 ArithDataset collate)
  power/area/delay
  node_powers/node_power_mask — 保留 (Route D 兼容)
  bit_width    — 新增标记 (训练时可作辅助 feature 或筛选)
"""
import argparse
import torch
from tqdm import tqdm


def slice_onehot(data, bit_width):
    """从原 X 切出 [:, 3:7] type onehot, 加 bit_width 标签"""
    out = []
    for item in tqdm(data, desc=f"slice {bit_width}-bit"):
        X_full = item["X"]
        # 切 4 维 type onehot (X[:, 3:7] 在 7d 和 13d 中位置一致)
        X_onehot = X_full[:, 3:7].clone().float()
        new_item = {
            "X": X_onehot,
            "edge_index": item["edge_index"].clone(),
            "edge_attr": item["edge_attr"].clone(),
            "area": float(item["area"]),
            "delay": float(item["delay"]),
            "power": float(item["power"]),
            "bit_width": bit_width,
        }
        # 保留 node_powers 兼容
        if "node_powers" in item:
            new_item["node_powers"] = item["node_powers"].clone()
            new_item["node_power_mask"] = item["node_power_mask"].clone()
        out.append(new_item)
    return out


def main(in_8bit, in_16bit, out_path):
    print(f"📂 加载 8-bit: {in_8bit}")
    d8 = torch.load(in_8bit, map_location="cpu", weights_only=False)
    print(f"   {len(d8)} 样本, X dim={d8[0]['X'].shape[1]}")

    print(f"📂 加载 16-bit: {in_16bit}")
    d16 = torch.load(in_16bit, map_location="cpu", weights_only=False)
    print(f"   {len(d16)} 样本, X dim={d16[0]['X'].shape[1]}")

    merged = slice_onehot(d8, 8) + slice_onehot(d16, 16)

    # 打乱顺序避免 8-bit 全在前面 (KFold 划分会受影响)
    import random
    rng = random.Random(42)
    rng.shuffle(merged)

    print(f"\n📊 合并后: {len(merged)} 样本")
    print(f"   X dim: {merged[0]['X'].shape[1]} (4 维 type onehot)")
    n_8 = sum(1 for x in merged if x['bit_width'] == 8)
    n_16 = sum(1 for x in merged if x['bit_width'] == 16)
    print(f"   8-bit: {n_8} ({n_8/len(merged)*100:.1f}%)")
    print(f"   16-bit: {n_16} ({n_16/len(merged)*100:.1f}%)")

    import statistics
    powers = [x['power'] for x in merged]
    print(f"   power: mean={statistics.mean(powers):.4f}, "
          f"std={statistics.stdev(powers):.4f}, "
          f"range [{min(powers):.4f}, {max(powers):.4f}]")

    print(f"\n💾 保存: {out_path}")
    torch.save(merged, out_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in_8bit", default="dataset/glitch_power_data_8bit_v2.pt")
    p.add_argument("--in_16bit", default="dataset/glitch_power_data_16bit_v2_13k_enriched.pt")
    p.add_argument("--out", default="dataset/glitch_power_data_mixed_onehot.pt")
    args = p.parse_args()
    main(args.in_8bit, args.in_16bit, args.out)
