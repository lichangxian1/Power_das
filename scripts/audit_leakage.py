"""近重复样本跨 fold 泄漏体检。

背景: 数据是"同一 16-bit PP 结构上随机采样 routing"。精确 graph-hash 去重
(generate_dataset.compute_graph_hash) 只能去掉完全相同的图; 两个 routing 仅差
一两条边的"近重复"哈希不同, 会被 KFold 分到不同 fold → val/test 里出现 train
的近邻 → 抬高 fix-N / per-N 指标。本脚本量化这种泄漏。

方法 (按 N 分桶, 同结构家族才可比):
  - 每个图的"身份" = 其有向边集合 {src*(N+1)+dst}。同 N 家族里 PP→stage0 等
    结构边对所有样本相同, 会在"对称差"里自动抵消, 因此对称差只反映 routing 差异。
  - 用稀疏 0/1 矩阵 M[图, 边], shared = M·Mᵀ 给出共享边数,
    对称差距离 dist(i,j) = E_i + E_j - 2·shared(i,j)。
  - 复现 train_proxy 的 KFold(shuffle=True, random_state=42) fold 划分,
    对每个图求其最近邻 (不含自身), 看最近邻是否落在不同 fold。

用法:
    python scripts/audit_leakage.py \
        --data dataset/glitch_power_data_16bit_v2_11k_edge10.pt \
        --n_folds 5 --focus_n 730
"""
import argparse

import numpy as np
import torch
from scipy import sparse
from sklearn.model_selection import KFold


def fold_assignment(n_samples, n_folds, seed=42):
    """复现 train_proxy.py 的 KFold: 返回 fold_of[i] = 样本 i 作为 val 的 fold 号。"""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_of = np.full(n_samples, -1, dtype=int)
    for fid, (_train_idx, val_idx) in enumerate(kf.split(range(n_samples))):
        fold_of[val_idx] = fid
    return fold_of


def edge_signatures(items, idxs):
    """对给定样本下标, 返回每个图的边整数集合 (list[np.ndarray])。"""
    sigs = []
    for i in idxs:
        ei = items[i]["edge_index"]
        n = items[i]["X"].shape[0]
        src = ei[0].long().numpy()
        dst = ei[1].long().numpy()
        codes = src.astype(np.int64) * (n + 1) + dst.astype(np.int64)
        sigs.append(np.unique(codes))
    return sigs


def build_matrix(sigs):
    """list[edge-code set] → 稀疏 0/1 矩阵 [图, 全局边]。"""
    vocab = {}
    rows, cols = [], []
    for r, codes in enumerate(sigs):
        for c in codes:
            j = vocab.get(c)
            if j is None:
                j = len(vocab)
                vocab[c] = j
            rows.append(r)
            cols.append(j)
    data = np.ones(len(rows), dtype=np.float32)
    M = sparse.csr_matrix((data, (rows, cols)),
                          shape=(len(sigs), len(vocab)), dtype=np.float32)
    return M


def audit_bucket(items, bucket_idxs, fold_of, name):
    g = len(bucket_idxs)
    if g < 2:
        print(f"  [{name}] 样本不足 ({g}), 跳过")
        return
    sigs = edge_signatures(items, bucket_idxs)
    E = np.array([len(s) for s in sigs], dtype=np.float64)
    M = build_matrix(sigs)

    shared = (M @ M.T).toarray().astype(np.float64)   # [g, g] 共享边数
    # 对称差距离 dist = E_i + E_j - 2*shared
    dist = E[:, None] + E[None, :] - 2.0 * shared
    np.fill_diagonal(dist, np.inf)

    nn_dist = dist.min(axis=1)
    nn_j = dist.argmin(axis=1)
    folds = fold_of[bucket_idxs]
    nn_cross = folds != folds[nn_j]          # 最近邻在不同 fold = 跨 fold 近邻
    avg_E = E.mean()

    print(f"\n  [{name}] 图数={g}, 平均边数≈{avg_E:.0f}")
    print(f"    最近邻对称差距离 (改变的连接数):")
    print(f"      min={nn_dist.min():.0f}  median={np.median(nn_dist):.0f}  "
          f"mean={nn_dist.mean():.1f}  max={nn_dist.max():.0f}")
    # 不同近似程度下的占比 + 其中跨 fold 比例
    print(f"    {'阈值(≤改变边)':<16}{'近邻占比':<12}{'其中跨fold(泄漏)':<16}")
    for thr in (0, 2, 4, 8, 16):
        near = nn_dist <= thr
        cnt = int(near.sum())
        frac = cnt / g
        cross_frac = float(nn_cross[near].mean()) if cnt > 0 else 0.0
        # 归一化: 改变边数占平均边数的比例
        pct = thr / avg_E * 100
        print(f"      ≤{thr:<3d}({pct:4.1f}%){'':<5}{frac*100:6.2f}% ({cnt})"
              f"{'':<3}{cross_frac*100:6.2f}%")

    # 总体: 最近邻落在不同 fold 的占比 (越高=KFold 越可能把近邻拆到 train/val)
    print(f"    任意最近邻跨 fold 占比: {nn_cross.mean()*100:.2f}%")
    # 危险样本: 近邻很近 (<=2) 且跨 fold → 直接泄漏到 val/test
    danger = (nn_dist <= 2) & nn_cross
    print(f"    ⚠️  高泄漏风险样本 (近邻≤2 且跨fold): {int(danger.sum())} "
          f"({danger.mean()*100:.2f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--focus_n", type=int, default=730,
                    help="重点体检的 N (fix-N 评估用的那个结构家族)")
    ap.add_argument("--top_buckets", type=int, default=3,
                    help="额外体检样本量最大的前几个 N 桶")
    args = ap.parse_args()

    print(f"📂 加载 {args.data}")
    items = torch.load(args.data, map_location="cpu", weights_only=False)
    n = len(items)
    print(f"   样本数={n}")

    fold_of = fold_assignment(n, args.n_folds, args.seed)
    ns = np.array([items[i]["X"].shape[0] for i in range(n)])

    uniq, counts = np.unique(ns, return_counts=True)
    order = np.argsort(-counts)
    print(f"   N 分布 (top): " +
          ", ".join(f"N={uniq[o]}:{counts[o]}" for o in order[:8]))

    # 1) 重点桶 focus_n
    focus_idxs = np.where(ns == args.focus_n)[0]
    audit_bucket(items, focus_idxs, fold_of, f"focus N={args.focus_n}")

    # 2) 样本量最大的几个桶
    for o in order[:args.top_buckets]:
        nval = int(uniq[o])
        if nval == args.focus_n:
            continue
        idxs = np.where(ns == nval)[0]
        audit_bucket(items, idxs, fold_of, f"N={nval}")

    print("\n说明: '跨 fold 近邻' = 某图的最近邻被 KFold 分到了别的 fold。")
    print("      KFold 是随机划分, 近邻天然约 (n_folds-1)/n_folds 落在不同 fold;")
    print("      真正的风险信号是 '近邻≤2 且跨fold' 的占比 —— 它直接量化了")
    print("      train 里有多少近重复样本泄漏进了某个 fold 的 val/test。")


if __name__ == "__main__":
    main()
