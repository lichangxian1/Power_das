"""
诊断 ArithProxyGNN 内部信号分布，回答三个问题：

  H1: sum_power 与 global_corr 谁主导？双头是否失衡？
  H2: GNN 输出 h_i 在多大程度上随路由变化？(跨样本方差 vs 样本内方差)
  H3: 节点级 node_power 输出是否真的有路由特异性？

用法:
    python3 trainer/diagnose_proxy.py --ckpt dataset/glitch_power_proxy_gnn_fold0.pth

如果没有 _fold0.pth，可以用最终的 glitch_power_proxy_gnn.pth。
"""
import os
import sys
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_proxy import ArithDataset, custom_collate
from proxy_mlp import ArithProxyGNN


# ============================================================
# 加载模型
# ============================================================
def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["model_state_dict"]

    # 从 state_dict 推断 use_edge_feat（无显式记录的旧 ckpt 也能加载）
    has_edge_proj = any("edge_proj" in k for k in sd.keys())
    use_edge_feat = ckpt.get("use_edge_feat", has_edge_proj)
    use_mean_agg = ckpt.get("use_mean_agg", True)

    model = ArithProxyGNN(
        node_feature_dim=ckpt["node_feature_dim"],
        hidden_dim=ckpt["hidden_dim"],
        num_gnn_layers=ckpt["num_gnn_layers"],
        dropout=0.0,
        use_mean_agg=use_mean_agg,
        use_edge_feat=use_edge_feat,
    ).to(device)
    model.load_state_dict(sd)
    model.eval()

    print(f"  ✅ 模型加载成功: {ckpt_path}")
    print(f"     hidden_dim={model.hidden_dim}, num_gnn_layers={model.num_gnn_layers}, "
          f"in_dim={model.node_feature_dim}")
    print(f"     use_mean_agg={use_mean_agg}, use_edge_feat={use_edge_feat}")
    if "best_tau" in ckpt:
        print(f"     训练时 best τ = {ckpt['best_tau']:+.4f}")
    return model, ckpt


# ============================================================
# Mirror of ArithProxyGNN.forward, 同时返回所有中间量
# ============================================================
@torch.no_grad()
def forward_with_intermediates(model, x_node, edge_index, mask):
    B, N, _ = x_node.shape
    H = model.hidden_dim
    x_flat = x_node.reshape(B * N, -1)
    mask_flat = mask.reshape(B * N)
    types_flat = x_flat[:, 3:7].argmax(dim=-1)
    h = model.input_proj(x_flat, types_flat, mask_flat)
    h = model.input_norm(h)

    edge_feat = None
    if model.use_edge_feat:
        src = edge_index[0]
        dst = edge_index[1]
        arrival_flat = x_flat[:, model.arrival_idx]
        edge_skew = arrival_flat[src] - arrival_flat[dst]
        edge_src_at = arrival_flat[src]
        edge_feat = torch.stack([edge_skew, edge_src_at], dim=-1)

    for layer in model.gnn_layers:
        h = layer(h, edge_index, edge_feat)

    mask_f = mask_flat.unsqueeze(-1).float()
    h_masked = h * mask_f
    node_power_flat = model.node_head(h_masked).squeeze(-1)
    node_power = node_power_flat.reshape(B, N) * mask.float()
    sum_power = node_power.sum(dim=1)

    h_3d = h_masked.reshape(B, N, H)
    n_real = mask.sum(dim=1, keepdim=True).float().clamp_min(1.0)
    mean_pool = h_3d.sum(dim=1) / n_real
    h_for_max = h_3d.masked_fill(~mask.unsqueeze(-1), float('-inf'))
    max_pool = h_for_max.max(dim=1)[0]
    max_pool = torch.where(torch.isinf(max_pool), torch.zeros_like(max_pool), max_pool)
    graph_feat = torch.cat([mean_pool, max_pool], dim=-1)
    global_corr = model.global_head(graph_feat).squeeze(-1)

    return {
        "pred": sum_power + global_corr,
        "sum_power": sum_power,
        "global_corr": global_corr,
        "h": h_3d,
        "node_power": node_power,
        "mask": mask,
    }


# ============================================================
# 诊断主函数
# ============================================================
def diagnose(data_path, ckpt_path, n_samples=256, batch_size=64, filter_n=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  🖥  Device: {device}")

    model, ckpt = load_model(ckpt_path, device)

    print(f"\n  📂 加载数据集: {data_path}")
    dataset = ArithDataset(data_path)

    n_per_sample = np.array([dataset.data[i]["X"].shape[0] for i in range(len(dataset))])

    # 用与训练一致的 KFold（seed=42）的 fold 0 验证集
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    _, val_idx = next(kf.split(range(len(dataset))))

    if filter_n is not None:
        # 只保留 N == filter_n 的样本，去除 N 异构带来的诊断噪声
        val_idx = [i for i in val_idx if n_per_sample[i] == filter_n]
        print(f"  🔍 过滤后只保留 N={filter_n} 的样本: {len(val_idx)} 个")
        if len(val_idx) < 16:
            print(f"  ❌ N={filter_n} 在 fold 0 val 中样本太少 ({len(val_idx)})")
            sys.exit(1)
        common_N = filter_n
    else:
        n_min, n_max = int(n_per_sample.min()), int(n_per_sample.max())
        if n_min != n_max:
            print(f"  ⚠️  样本节点数不一致: min={n_min}, max={n_max}, "
                  f"将对齐到 N={n_min} (截断)")
            print(f"      建议加 --filter_n 730 以避免 N 混杂噪声")
        else:
            print(f"  ✅ 所有样本节点数一致: N={n_min}")
        common_N = n_min

    val_idx = list(val_idx)[:n_samples]
    print(f"  📊 使用 fold 0 验证集 {len(val_idx)} 个样本做诊断 (common_N={common_N})")

    val_subset = Subset(dataset, val_idx)
    loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                        collate_fn=custom_collate)

    sums, corrs, preds, trues = [], [], [], []
    h_all = []          # list of [b, common_N, H]
    node_pow_all = []   # list of [b, common_N]

    for X, edge_index, mask, pn, pr in loader:
        X = X.to(device); edge_index = edge_index.to(device); mask = mask.to(device)
        out = forward_with_intermediates(model, X, edge_index, mask)
        sums.append(out["sum_power"].cpu())
        corrs.append(out["global_corr"].cpu())
        preds.append(out["pred"].cpu())
        trues.append(pn)
        h_all.append(out["h"][:, :common_N, :].cpu())
        node_pow_all.append(out["node_power"][:, :common_N].cpu())

    sums = torch.cat(sums)
    corrs = torch.cat(corrs)
    preds = torch.cat(preds)
    trues = torch.cat(trues)
    h = torch.cat(h_all, dim=0)              # [n, common_N, H]
    node_pow = torch.cat(node_pow_all, dim=0)  # [n, common_N]
    n = sums.size(0)

    # ============================================================
    # H0: Ranking metrics (R@K, τ, ρ) on this subset
    # ============================================================
    import scipy.stats as stats
    print(f"\n{'='*70}")
    print(f"  H0: Ranking 指标 (n={n} 样本, common_N={common_N})")
    print(f"{'='*70}")
    p_np = preds.numpy(); t_np = trues.numpy()
    tau = stats.kendalltau(p_np, t_np).correlation
    rho = stats.spearmanr(p_np, t_np).correlation
    pear = float(np.corrcoef(p_np, t_np)[0, 1])
    print(f"  Pearson  = {pear:+.4f}")
    print(f"  Kendall τ= {tau:+.4f}")
    print(f"  Spearman ρ= {rho:+.4f}")
    for r in (0.05, 0.10, 0.20):
        k = max(1, int(round(n * r)))
        ti = set(torch.topk(trues, k, largest=False).indices.tolist())
        pi = set(torch.topk(preds, k, largest=False).indices.tolist())
        print(f"  R@{int(r*100)}% = {len(ti & pi)/k:.3f}  (k={k})")

    # ============================================================
    # H1: 双头量级
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  H1: 双头量级 (归一化空间, n={n} 样本)")
    print(f"{'='*70}")
    print(f"  sum_power:   mean={sums.mean():+.4f}  std={sums.std():.4f}  "
          f"|max|={sums.abs().max():.4f}")
    print(f"  global_corr: mean={corrs.mean():+.4f}  std={corrs.std():.4f}  "
          f"|max|={corrs.abs().max():.4f}")
    print(f"  pred:        mean={preds.mean():+.4f}  std={preds.std():.4f}")
    print(f"  true:        mean={trues.mean():+.4f}  std={trues.std():.4f}")
    print(f"")
    r_sum_pred = (sums.std() / preds.std()).item()
    r_corr_pred = (corrs.std() / preds.std()).item()
    r_corr_sum = (corrs.std() / sums.std()).item()
    print(f"  std(sum)  / std(pred) = {r_sum_pred:.3f}")
    print(f"  std(corr) / std(pred) = {r_corr_pred:.3f}")
    print(f"  std(corr) / std(sum)  = {r_corr_sum:.4f}")
    print(f"")
    cor_sum_true = torch.corrcoef(torch.stack([sums, trues]))[0, 1].item()
    cor_corr_true = torch.corrcoef(torch.stack([corrs, trues]))[0, 1].item()
    cor_pred_true = torch.corrcoef(torch.stack([preds, trues]))[0, 1].item()
    print(f"  Pearson(sum_power,   true) = {cor_sum_true:+.4f}")
    print(f"  Pearson(global_corr, true) = {cor_corr_true:+.4f}")
    print(f"  Pearson(pred,        true) = {cor_pred_true:+.4f}")
    print(f"")
    print(f"  📌 H1 结论:")
    if r_corr_sum < 0.10:
        print(f"     双头失衡 — global_corr 变异 < 10% of sum_power")
        print(f"     模型实际上只用 sum_power 在做预测，global_head 是死参数")
    elif r_corr_sum < 0.30:
        print(f"     修正头有贡献但不主导 (corr/sum = {r_corr_sum:.2f})")
    else:
        print(f"     两头都活跃 (corr/sum = {r_corr_sum:.2f})")

    # ============================================================
    # H2: GNN 输出 h 的跨样本 vs 样本内方差
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  H2: GNN 输出 h_i 的跨样本 vs 样本内方差")
    print(f"{'='*70}")
    # h: [n, N, H]
    cross_var_h = h.var(dim=0).mean().item()         # 每个 (node_pos, hidden_dim) 跨 n 样本算 var，平均
    within_var_h = h.var(dim=1).mean().item()        # 每个样本跨节点位置算 var，平均
    ratio_h = cross_var_h / (within_var_h + 1e-12)

    print(f"  h 跨样本方差 (固定节点位置, 看路由差异让 h 变了多少):")
    print(f"     mean = {cross_var_h:.6e}")
    print(f"  h 样本内方差 (固定样本, 看不同节点位置 h 差异):")
    print(f"     mean = {within_var_h:.6e}")
    print(f"  比值 (跨样本 / 样本内) = {ratio_h:.4f}")
    print(f"")
    print(f"  📌 H2 结论:")
    if ratio_h < 0.02:
        print(f"     h_i 几乎完全由节点静态特征决定 (跨样本变异 < 2%)")
        print(f"     → GNN 几乎没把路由信息编码进 h")
    elif ratio_h < 0.10:
        print(f"     h_i 主要由静态特征决定 (跨样本变异 {ratio_h*100:.1f}%)")
        print(f"     → GNN 编码了少量路由差异，但很弱")
    else:
        print(f"     h_i 对路由有明显响应 ({ratio_h*100:.1f}%)")

    # ============================================================
    # H3: node_power 输出的跨样本 vs 样本内方差
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  H3: 节点级 node_power 输出的跨样本 vs 样本内方差")
    print(f"{'='*70}")
    cross_var_np = node_pow.var(dim=0)             # [common_N]
    within_var_np = node_pow.var(dim=1)            # [n]
    ratio_np = (cross_var_np.mean() / (within_var_np.mean() + 1e-12)).item()

    print(f"  node_power 跨样本方差 (固定节点位置):")
    print(f"     mean = {cross_var_np.mean():.6e}, max = {cross_var_np.max():.6e}")
    print(f"  node_power 样本内方差 (固定样本):")
    print(f"     mean = {within_var_np.mean():.6e}, max = {within_var_np.max():.6e}")
    print(f"  比值 (跨样本 / 样本内) = {ratio_np:.4f}")

    # 找跨样本变异最大的节点位置（路由敏感节点）
    topk_var, topk_idx = cross_var_np.topk(min(10, common_N))
    mean_per_pos = node_pow.mean(dim=0)
    print(f"\n  跨样本 var 最大的 10 个节点 (路由敏感节点):")
    print(f"    rank | node_pos | cross-var       | mean(node_pow)")
    for rank, (v, i) in enumerate(zip(topk_var.tolist(), topk_idx.tolist())):
        print(f"    #{rank+1:2d}   | {i:5d}    | {v:.6e}    | {mean_per_pos[i]:+.4f}")

    # 累积贡献：前 K 个节点贡献多少跨样本方差
    sorted_var, _ = cross_var_np.sort(descending=True)
    total = sorted_var.sum().item()
    cumsum = torch.cumsum(sorted_var, dim=0)
    print(f"\n  累积方差贡献:")
    for pct in [0.5, 0.8, 0.95]:
        idxs = (cumsum >= total * pct).nonzero(as_tuple=True)[0]
        if len(idxs):
            n_needed = idxs[0].item() + 1
            print(f"    前 {n_needed:4d} 个节点 ({n_needed/common_N*100:5.1f}% of all) "
                  f"贡献 {pct*100:.0f}% 方差")
    print(f"")
    print(f"  📌 H3 结论:")
    if ratio_np < 0.02:
        print(f"     node_power 几乎不随路由变化 (跨样本变异 < 2%)")
        print(f"     → 模型对不同路由几乎给出相同输出")
    elif ratio_np < 0.10:
        print(f"     node_power 对路由响应较弱 ({ratio_np*100:.1f}%)")
    else:
        print(f"     node_power 对路由有显著响应 ({ratio_np*100:.1f}%)")

    # ============================================================
    # 综合结论
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  🎯 综合结论")
    print(f"{'='*70}")
    print(f"  H1: std(corr) / std(sum)         = {r_corr_sum:.3f}  "
          + ("← 双头失衡" if r_corr_sum < 0.10 else "← 正常"))
    print(f"  H2: cross/within var on h        = {ratio_h:.3f}  "
          + ("← GNN 对路由不敏感" if ratio_h < 0.05 else "← GNN 编码了路由"))
    print(f"  H3: cross/within var on node_pow = {ratio_np:.3f}  "
          + ("← 节点输出对路由不敏感" if ratio_np < 0.05 else "← 节点输出对路由响应"))
    print(f"\n  说明:")
    print(f"    cross-sample var: 固定节点位置, 跨样本算方差 (代表路由变化让模型输出多大不同)")
    print(f"    within-sample var: 固定样本, 跨节点位置算方差 (代表静态结构差异)")
    print(f"    比值 ~ 1 表示 GNN 对路由敏感; 比值 << 1 表示 GNN 对路由不敏感")


# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset/glitch_power_data_16bit_enriched.pt")
    parser.add_argument("--ckpt", default="dataset/glitch_power_proxy_gnn_fold0.pth")
    parser.add_argument("--n_samples", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--filter_n", type=int, default=None,
                        help="只用 X.shape[0]==filter_n 的样本做诊断，去除 N 异构噪声")
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        print(f"  ❌ ckpt 不存在: {args.ckpt}")
        print(f"  提示: 跑 `python3 trainer/train_proxy.py` 让它完成至少 fold 0,")
        print(f"        训练脚本会自动保存 fold0.pth 到 {args.ckpt}")
        sys.exit(1)

    diagnose(args.data, args.ckpt, args.n_samples, args.batch_size, args.filter_n)
