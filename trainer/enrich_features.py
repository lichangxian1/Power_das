"""
特征增强 v2：在已有 9 维（含 arrival_time + input_skew）基础上再加 4 维图拓扑特征
X: [N, 9]  →  X: [N, 13]

新增特征（按顺序）:
  9.  fanout       — 节点出边数（驱动多少个下游）
  10. fanin        — 节点入边数（被多少个上游驱动）
  11. depth_to_pp  — 节点到最近 PP 的最长路径深度（拓扑深度）
  12. is_critical  — 是否在最长路径上 (0/1)

这些都是**图算法可直接计算**的物理量，不需要重跑 EDA。
基于稀疏 edge_index 实现，O(N+E) per sample。

用法:
    python3 trainer/enrich_features.py
"""

import os
import sys
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ========================== 单样本特征计算 ==========================
def compute_graph_features(X, edge_index):
    """
    Args:
        X: [N, 9]  现有特征（前 7 维含 type one-hot at idx 3..7）
        edge_index: [2, E]  稀疏边表，edge_index[0]=src, edge_index[1]=dst
    Returns:
        fanout:      [N]  float
        fanin:       [N]  float
        depth_to_pp: [N]  float
        is_critical: [N]  float (0/1)
    """
    N = X.shape[0]
    src = edge_index[0].long()
    dst = edge_index[1].long()

    # ---- fanout / fanin ----
    fanout = torch.zeros(N, dtype=torch.float32)
    fanin = torch.zeros(N, dtype=torch.float32)
    fanout.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float32))
    fanin.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))

    # ---- depth_to_pp: 拓扑序上的最长路径 ----
    # PP 节点 (type=2) 深度=0；其它节点 = max(前驱深度) + 1
    node_types = X[:, 3:7].argmax(dim=1)  # [N]
    is_pp = (node_types == 2)

    # 按 stage_idx 升序作为拓扑序近似（PP 强制最前）
    stage = X[:, 0].clone().float()
    stage[is_pp] = -1e9
    order = torch.argsort(stage, stable=True)

    # 构建邻接表（按 dst 索引）：preds[v] = list of u with u->v
    preds = [[] for _ in range(N)]
    for s, d in zip(src.tolist(), dst.tolist()):
        if 0 <= d < N:
            preds[d].append(s)

    depth = torch.zeros(N, dtype=torch.float32)
    for v_t in order:
        v = v_t.item()
        if is_pp[v]:
            depth[v] = 0.0
            continue
        p_list = preds[v]
        if p_list:
            depth[v] = depth[torch.tensor(p_list, dtype=torch.long)].max() + 1
        else:
            depth[v] = 0.0

    # ---- is_critical: 反向回溯最长路径 ----
    # 1) 找深度最大的"输出"节点（这里取全图深度最大者作为终点集合）
    max_depth = depth.max().item()
    is_critical = torch.zeros(N, dtype=torch.float32)
    if max_depth > 0:
        # 从所有 depth==max_depth 的节点反向走，每步选 depth=cur-1 的前驱
        frontier = set((depth == max_depth).nonzero(as_tuple=True)[0].tolist())
        is_critical[list(frontier)] = 1.0
        # 反向 BFS
        seen = set(frontier)
        while frontier:
            new_frontier = set()
            for v in frontier:
                cur_d = depth[v].item()
                if cur_d <= 0:
                    continue
                for u in preds[v]:
                    if u in seen:
                        continue
                    if depth[u].item() == cur_d - 1:
                        new_frontier.add(u)
            for u in new_frontier:
                is_critical[u] = 1.0
                seen.add(u)
            frontier = new_frontier

    return fanout, fanin, depth, is_critical


# ========================== 数据集增强 ==========================
def enrich_dataset(in_path, out_path, force=False):
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"找不到输入文件: {in_path}")

    data = torch.load(in_path, map_location="cpu")
    print(f"  📂 加载 {len(data)} 个样本: {in_path}")

    first = data[0]
    if "edge_index" not in first:
        raise ValueError("数据集必须是 edge_index 格式（不支持旧 P 矩阵格式，请先转换）")

    feat_dim = first["X"].shape[1]
    if feat_dim >= 13 and not force:
        print(f"  ⚠️ 检测到 X 已有 {feat_dim} 维特征，似乎已经增强过")
        print(f"     如需强制覆盖请传 force=True")
        return
    if feat_dim != 9:
        print(f"  ⚠️ 期望输入 X 是 9 维（arrival+skew 增强后），实际 {feat_dim} 维")
        print(f"     脚本会从前 9 维开始扩展，请确认这是你想要的")

    all_fanout, all_fanin, all_depth, all_crit = [], [], [], []
    fail_count = 0

    for item in tqdm(data, desc="计算图拓扑特征"):
        try:
            X = item["X"]
            edge_index = item["edge_index"]

            fo, fi, dp, cr = compute_graph_features(X, edge_index)

            # 从原 9 维基础上扩展（防累加）
            X_base = X[:, :9]
            X_new = torch.cat([
                X_base,
                fo.unsqueeze(1),
                fi.unsqueeze(1),
                dp.unsqueeze(1),
                cr.unsqueeze(1),
            ], dim=1)
            item["X"] = X_new

            all_fanout.append(fo)
            all_fanin.append(fi)
            all_depth.append(dp)
            all_crit.append(cr)
        except Exception as e:
            fail_count += 1
            if fail_count <= 5:
                print(f"  ⚠️ 样本处理失败: {e}")

    # 统计
    all_fanout = torch.cat(all_fanout)
    all_fanin = torch.cat(all_fanin)
    all_depth = torch.cat(all_depth)
    all_crit = torch.cat(all_crit)

    print()
    print(f"  📊 新增特征分布:")
    print(f"     fanout:       mean={all_fanout.mean():.2f}  max={all_fanout.max():.0f}  "
          f"std={all_fanout.std():.2f}")
    print(f"     fanin:        mean={all_fanin.mean():.2f}  max={all_fanin.max():.0f}  "
          f"std={all_fanin.std():.2f}")
    print(f"     depth_to_pp:  mean={all_depth.mean():.2f}  max={all_depth.max():.0f}  "
          f"std={all_depth.std():.2f}")
    print(f"     is_critical:  mean={all_crit.mean():.4f}  "
          f"(关键路径节点占比 {all_crit.mean()*100:.2f}%)")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(data, out_path)
    print()
    print(f"  💾 已保存: {out_path}")
    print(f"     新 X 维度: {list(data[0]['X'].shape)}")
    print(f"     失败样本: {fail_count}")


if __name__ == "__main__":
    enrich_dataset(
        in_path="dataset/glitch_power_data_16bit_merged.pt",
        out_path="dataset/glitch_power_data_16bit_enriched.pt",
    )
