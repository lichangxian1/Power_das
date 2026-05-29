"""
把 Route D 原始数据集 (X[N,7] + edge_index + node_powers) enrich 到 X[N,13]，
保留 node_powers / node_power_mask 字段。

链路: 7 → 9 (timing) → 13 (graph topology)
"""
import os
import sys
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trainer"))
from augment_features import compute_timing_features, DEFAULT_TYPE_DELAYS
from enrich_features import compute_graph_features


def edge_index_to_P(edge_index, N):
    P = torch.zeros(N, N, dtype=torch.float32)
    src = edge_index[0].long()
    dst = edge_index[1].long()
    P[src, dst] = 1.0
    return P


def main(in_path, out_path):
    data = torch.load(in_path, map_location="cpu", weights_only=False)
    print(f"  📂 加载 {len(data)} 样本: {in_path}")
    print(f"     原始 X 维度: {data[0]['X'].shape}")
    print(f"     含 node_powers: {bool(data[0].get('node_power_mask') is not None)}")

    for item in tqdm(data, desc="enrich 7→13"):
        X = item["X"]
        N = X.shape[0]
        if "edge_index" in item:
            edge_index = item["edge_index"]
            P = edge_index_to_P(edge_index, N)
        else:
            P = item["P"].float()
            edge_index = P.nonzero(as_tuple=False).t().contiguous().long()
            item["edge_index"] = edge_index
            del item["P"]

        # 7 → 9: timing features
        arrival, skew = compute_timing_features(X[:, :7], P, DEFAULT_TYPE_DELAYS)
        X9 = torch.cat([X[:, :7], arrival.unsqueeze(1), skew.unsqueeze(1)], dim=1)

        # 9 → 13: graph topology features
        fo, fi, dp, cr = compute_graph_features(X9, edge_index)
        X13 = torch.cat([
            X9, fo.unsqueeze(1), fi.unsqueeze(1),
            dp.unsqueeze(1), cr.unsqueeze(1),
        ], dim=1)
        item["X"] = X13

    print(f"\n  ✅ enrich 完成。新 X 维度: {data[0]['X'].shape}")
    print(f"  💾 保存到: {out_path}")
    torch.save(data, out_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", default="dataset/glitch_power_data_16bit_node_power.pt")
    parser.add_argument("--out_path", default="dataset/glitch_power_data_16bit_node_power_enriched.pt")
    args = parser.parse_args()
    main(args.in_path, args.out_path)
