"""合并 v2_a.pt + v2_b.pt → v2.pt (按 graph_hash dedup)

用法:
    python scripts/merge_v2_ab.py
        # 默认: v2_a.pt + v2_b.pt → v2.pt
    python scripts/merge_v2_ab.py --out v2_merged.pt
"""
import os, sys
import argparse
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.generate_dataset import compute_graph_hash


def main(in_a, in_b, out):
    print(f"📂 加载 A: {in_a}")
    a = torch.load(in_a, map_location="cpu", weights_only=False)
    print(f"   {len(a)} 样本")
    print(f"📂 加载 B: {in_b}")
    b = torch.load(in_b, map_location="cpu", weights_only=False)
    print(f"   {len(b)} 样本")

    seen = {}
    out_list = []
    dup_a = 0
    dup_b = 0

    for d in tqdm(a, desc="A"):
        h = compute_graph_hash(d["X"], d["edge_index"], d["edge_attr"])
        if h in seen:
            dup_a += 1
            continue
        seen[h] = True
        out_list.append(d)

    for d in tqdm(b, desc="B"):
        h = compute_graph_hash(d["X"], d["edge_index"], d["edge_attr"])
        if h in seen:
            dup_b += 1
            continue
        seen[h] = True
        out_list.append(d)

    print(f"\n📊 A: {len(a)} → 去重 {dup_a}")
    print(f"📊 B: {len(b)} → 去重 {dup_b}")
    print(f"📊 合并后唯一样本: {len(out_list)}")
    print(f"💾 保存到: {out}")
    torch.save(out_list, out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in_a", default="dataset/glitch_power_data_16bit_v2_a.pt")
    p.add_argument("--in_b", default="dataset/glitch_power_data_16bit_v2_b.pt")
    p.add_argument("--out", default="dataset/glitch_power_data_16bit_v2.pt")
    args = p.parse_args()
    main(args.in_a, args.in_b, args.out)
