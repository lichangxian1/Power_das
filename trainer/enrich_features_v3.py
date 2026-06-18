"""特征增强 v3：在已有 13 维基础上追加 3 维「反向/关键路径」时序特征。

X: [N, 13]  →  X: [N, 16]

现有 13 维 (idx):
  0 stage_idx | 1 col_idx | 2 idx | 3-6 type one-hot | 7 arrival | 8 skew
  9 fanout | 10 fanin | 11 depth_to_pp(正向最长深度) | 12 is_critical(0/1)

新增 3 维 (按顺序追加, 全部基于加权 arrival, 在规则阵列上仍有区分度):
  13. rev_arrival   — 节点到主输出的最长「加权」路径 (reverse arrival / 类 required)
  14. slack         — 全图加权关键路径长 - 经过该节点的最长加权路径, ≥0
                      =0 即落在加权关键路径上 (标准 STA slack, 物理最相关)
  15. arrival_slack — max_arrival - arrival (该节点离最晚到达的差)

动机: 现有 13 维只有“输入侧正向”的拓扑量, 缺“到输出侧”的反向时序。
delay 由完整加权关键路径 (输入->输出) 决定, 反向 slack 能提升同规模(Fixed-N)区分力。
注: unweighted 路径长在规则乘法/加法阵列上处处相等(退化), 故全部改用加权 arrival。
全部是 O(N+E) 图算法, 不需重跑 EDA。

用法:
    python3 trainer/enrich_features_v3.py \
        --in dataset/glitch_power_data_16bit_v2_11k_edge10.pt \
        --out dataset/glitch_power_data_16bit_v2_11k_edge10_cp.pt
"""

import argparse
import os
import sys

import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ARRIVAL_IDX = 7
DEPTH_FWD_IDX = 11


def compute_critical_path_features(X, edge_index):
    """返回 (rev_arrival, slack, arrival_slack)，均为 [N] float32。"""
    N = X.shape[0]
    src = edge_index[0].long()
    dst = edge_index[1].long()
    arrival = X[:, ARRIVAL_IDX].float()

    # 节点自身延迟 d[v] = arrival[v] - max(前驱 arrival); 源节点 d=arrival
    maxpred = torch.full((N,), -1.0e9)
    maxpred.scatter_reduce_(0, dst, arrival[src], reduce="amax", include_self=True)
    is_src = maxpred < -1.0e8
    dgate = torch.where(is_src, arrival, arrival - maxpred).clamp_min(0.0)

    # 后继邻接表
    succs = [[] for _ in range(N)]
    for s, d in zip(src.tolist(), dst.tolist()):
        if 0 <= s < N:
            succs[s].append(d)

    # rev_arrival[v] = d[v] + max(后继 rev_arrival); 汇节点 rev=d[v]
    # 反向拓扑序: stage 升序的逆序 (输出节点先处理)
    order = torch.argsort(X[:, 0].float(), stable=True).tolist()
    rev_arrival = torch.zeros(N, dtype=torch.float32)
    for v in reversed(order):
        sl = succs[v]
        if sl:
            rev_arrival[v] = dgate[v] + rev_arrival[torch.tensor(sl, dtype=torch.long)].max()
        else:
            rev_arrival[v] = dgate[v]

    # 经过节点的最长加权路径 (arrival 与 rev_arrival 都含 d[v], 相加减一次)
    path_arrival = arrival + rev_arrival - dgate
    g_crit = path_arrival.max().clamp_min(1e-6)
    slack = (g_crit - path_arrival).clamp_min(0.0)

    arrival_slack = arrival.max() - arrival
    return rev_arrival, slack, arrival_slack


def enrich_dataset(in_path, out_path, limit=None):
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"找不到输入文件: {in_path}")

    raw = torch.load(in_path, map_location="cpu")
    data = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
    print(f"  📂 加载 {len(data)} 个样本: {in_path}")

    feat_dim = data[0]["X"].shape[1]
    if feat_dim != 13:
        print(f"  ⚠️ 期望输入 X 是 13 维, 实际 {feat_dim} 维 —— 脚本按“前13维保留+追加”处理")

    all_rev, all_slk, all_aslk = [], [], []
    fail = 0
    items = data if limit is None else data[:limit]
    for item in tqdm(items, desc="计算关键路径特征"):
        try:
            X = item["X"]
            rev, slk, aslk = compute_critical_path_features(X, item["edge_index"])
            item["X"] = torch.cat([
                X[:, :13],
                rev.unsqueeze(1), slk.unsqueeze(1), aslk.unsqueeze(1),
            ], dim=1)
            all_rev.append(rev); all_slk.append(slk); all_aslk.append(aslk)
        except Exception as e:  # noqa: BLE001
            fail += 1
            if fail <= 5:
                print(f"  ⚠️ 样本失败: {e}")

    all_rev = torch.cat(all_rev); all_slk = torch.cat(all_slk); all_aslk = torch.cat(all_aslk)
    print()
    print("  📊 新增特征分布:")
    print(f"     rev_arrival:   mean={all_rev.mean():.2f} max={all_rev.max():.1f} std={all_rev.std():.2f} "
          f"唯一值≈{torch.unique(all_rev).numel()}")
    print(f"     slack:         mean={all_slk.mean():.3f} max={all_slk.max():.1f} std={all_slk.std():.3f} "
          f"(=0 关键路径节点占比 {(all_slk <= 1e-6).float().mean()*100:.2f}%)")
    print(f"     arrival_slack: mean={all_aslk.mean():.3f} max={all_aslk.max():.1f} std={all_aslk.std():.3f}")
    print(f"     NaN 检查: rev={torch.isnan(all_rev).any().item()} "
          f"slk={torch.isnan(all_slk).any().item()} aslk={torch.isnan(all_aslk).any().item()}")

    if limit is not None:
        print(f"\n  🧪 limit={limit} 小样本验证模式, 不保存")
        return

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(raw if isinstance(raw, dict) else data, out_path)
    print(f"\n  💾 已保存: {out_path}")
    print(f"     新 X 维度: {list(data[0]['X'].shape)}  失败样本: {fail}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path",
                    default="dataset/glitch_power_data_16bit_v2_11k_edge10.pt")
    ap.add_argument("--out", dest="out_path",
                    default="dataset/glitch_power_data_16bit_v2_11k_edge10_cp.pt")
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 个样本做验证, 不保存")
    args = ap.parse_args()
    enrich_dataset(args.in_path, args.out_path, limit=args.limit)
