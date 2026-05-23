"""
合并多个旧格式 dataset 文件，统一转换为新格式

支持的输入格式（自动识别）:
  - X[N, 7]  + P[N, N]            (最原始格式)
  - X[N, 9]  + P[N, N]            (augment_features 跑过的中间格式，比如你的 v2)
  - X[N, 9]  + edge_index[2, E]   (已经是新格式)

输出格式:
  - X[N, 9]  + edge_index[2, E]
  - 含 area, delay, power 标签
  - 已基于图结构哈希做跨文件去重

用法: 修改文件末尾的 in_paths / out_path 即可
"""

import os
import torch
import hashlib
from tqdm import tqdm


# ========================== 时序特征计算 ==========================
DEFAULT_TYPE_DELAYS = (2.0, 1.0, 0.0, 0.5)


def compute_timing_features(X_base, edge_index, type_delays=DEFAULT_TYPE_DELAYS):
    """从 X_base[N, 7] 和 edge_index[2, E] 计算 arrival_time 和 input_skew"""
    N = X_base.shape[0]
    type_delays_t = torch.tensor(type_delays, dtype=torch.float32)
    node_types = X_base[:, 3:7].argmax(dim=1)
    own_delays = type_delays_t[node_types]

    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    pred_lists = [[] for _ in range(N)]
    for s, d in zip(src, dst):
        pred_lists[d].append(s)

    sort_key = X_base[:, 0].clone().float()
    sort_key[node_types == 2] = -1e9
    order = torch.argsort(sort_key, stable=True)

    arrival = torch.zeros(N, dtype=torch.float32)
    for v_t in order:
        v = v_t.item()
        if node_types[v] == 2:
            arrival[v] = 0.0
            continue
        preds = pred_lists[v]
        if preds:
            pa = arrival[torch.tensor(preds, dtype=torch.long)]
            arrival[v] = pa.max() + own_delays[v]
        else:
            arrival[v] = own_delays[v]

    skew = torch.zeros(N, dtype=torch.float32)
    for v in range(N):
        preds = pred_lists[v]
        if len(preds) >= 2:
            pa = arrival[torch.tensor(preds, dtype=torch.long)]
            skew[v] = (pa.max() - pa.min()).item()

    return arrival, skew


# ========================== 图哈希（去重用） ==========================
def compute_graph_hash(X, edge_index):
    """基于 X (前 7 维) 和 排序后的 edge_index 计算 MD5 哈希"""
    X_struct = X[:, :7].numpy().tobytes()
    ei_np = edge_index.numpy()
    # 把边规范化排序（src*MAX + dst），消除插入顺序的影响
    flat = ei_np[0].astype('int64') * 100000 + ei_np[1].astype('int64')
    flat.sort()
    return hashlib.md5(X_struct + flat.tobytes()).hexdigest()


# ========================== 单项规范化 ==========================
def normalize_item(item):
    """
    把任意旧/新格式的 item 标准化为:
        {"X": X[N,9], "edge_index": ei[2,E], "area", "delay", "power"}
    """
    # 1) 邻接结构: P → edge_index
    if "edge_index" in item:
        edge_index = item["edge_index"].long()
    elif "P" in item:
        P = item["P"]
        edge_index = P.nonzero(as_tuple=False).t().contiguous().long()
    else:
        raise KeyError("item 既没有 edge_index 也没有 P")

    # 2) X: 7 维 → 9 维 (补 timing 特征)
    X = item["X"]
    if X.shape[1] == 7:
        arrival, skew = compute_timing_features(X, edge_index)
        X = torch.cat([X, arrival.unsqueeze(1), skew.unsqueeze(1)], dim=1)
    elif X.shape[1] != 9:
        raise ValueError(f"X 维度异常: {X.shape}，期望 7 或 9")

    return {
        "X": X,
        "edge_index": edge_index,
        "area":  item["area"],
        "delay": item["delay"],
        "power": item["power"],
    }


# ========================== 主合并函数 ==========================
def merge_datasets(in_paths, out_path):
    if isinstance(in_paths, str):
        in_paths = [in_paths]

    print(f"  🔗 准备合并 {len(in_paths)} 个数据集")
    for p in in_paths:
        print(f"     - {p}")
    print()

    merged = []
    seen_hashes = set()
    per_file_stats = []
    total_dup = 0

    for path in in_paths:
        if not os.path.exists(path):
            print(f"  ⚠️ 文件不存在，跳过: {path}")
            continue

        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            # 老版本 torch 不支持 weights_only 参数
            data = torch.load(path, map_location="cpu")

        print(f"  📂 加载 {len(data)} 条 from {os.path.basename(path)}")

        added = 0
        dup_in_file = 0
        err_in_file = 0

        for item in tqdm(data, desc=f"  规范化 {os.path.basename(path)}", leave=False):
            try:
                norm = normalize_item(item)
                h = compute_graph_hash(norm["X"], norm["edge_index"])
                if h in seen_hashes:
                    dup_in_file += 1
                    total_dup += 1
                    continue
                seen_hashes.add(h)
                merged.append(norm)
                added += 1
            except Exception as e:
                err_in_file += 1
                if err_in_file <= 3:
                    print(f"  ⚠️ 跳过一条样本: {e}")

        per_file_stats.append({
            "path": path,
            "total": len(data),
            "added": added,
            "dup":   dup_in_file,
            "err":   err_in_file,
        })
        print(f"     新增 {added} | 跨文件去重 {dup_in_file} | 异常 {err_in_file}\n")

    # 输出
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(merged, out_path)

    # 汇总报告
    print(f"\n  {'='*64}")
    print(f"  ✅ 合并完成")
    print(f"  {'='*64}")
    for s in per_file_stats:
        print(f"     {os.path.basename(s['path']):<45} {s['total']:>6} → +{s['added']:<6} (重复 {s['dup']}, 异常 {s['err']})")
    print(f"     {'─'*60}")
    print(f"     总计入库: {len(merged)} 条")
    print(f"     总去重:   {total_dup} 条")
    print(f"     输出路径: {out_path}")

    if len(merged) > 0:
        powers = [d["power"] for d in merged]
        delays = [d["delay"] for d in merged]
        areas  = [d["area"]  for d in merged]
        edges  = [d["edge_index"].shape[1] for d in merged]
        print(f"\n     标签统计:")
        print(f"       Power: mean={sum(powers)/len(powers):.6f}  min={min(powers):.6f}  max={max(powers):.6f}")
        print(f"       Delay: min={min(delays):.6f}  max={max(delays):.6f}")
        print(f"       Area : min={min(areas):.4f}  max={max(areas):.4f}")

        print(f"\n     格式校验:")
        print(f"       X shape:    {list(merged[0]['X'].shape)}  (期望 [N, 9])")
        print(f"       edge_index: {list(merged[0]['edge_index'].shape)}  (期望 [2, E])")
        print(f"       平均边数:    {sum(edges)/len(edges):.0f}")
    print(f"  {'='*64}")


if __name__ == "__main__":
    # === 在这里配置你要合并的文件 ===
    merge_datasets(
        in_paths=[
            "dataset/glitch_power_data_16bit.pt",       # 第 1 个老数据集
            "dataset/glitch_power_data_16bit_v3.pt",    # 第 2 个老数据集
            # 可以继续加更多文件
        ],
        out_path="dataset/glitch_power_data_16bit_merged.pt",
    )