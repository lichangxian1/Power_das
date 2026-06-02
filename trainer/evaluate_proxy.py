"""评估 ArithProxyGNN ckpt: mixed-N + fixed-N 指标 + 散点图

用法:
    # 评估单个 fold ckpt
    python trainer/evaluate_proxy.py \
        --ckpt dataset/glitch_power_proxy_gnn_B_v2_7k_e10_m234_fold0.pth \
        --data dataset/glitch_power_data_16bit_v2_7k_edge10.pt

    # 评估所有 fold (汇总 5-fold 平均)
    python trainer/evaluate_proxy.py \
        --ckpt_glob "dataset/glitch_power_proxy_gnn_B_v2_7k_e10_m234_fold*.pth" \
        --data dataset/glitch_power_data_16bit_v2_7k_edge10.pt

    # 自定义 val 划分 (用 KFold 复现训练时的 val set)
    python trainer/evaluate_proxy.py --ckpt ... --kfold 5 --fold_id 0
"""
import os
import sys
import argparse
import glob

import numpy as np
import torch
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_proxy import ArithDataset, custom_collate
from proxy_mlp import ArithProxyGNN, PureGIN, PureGCN, OneHotGIN


def _topk_recall(pred, true, k_ratios=(0.05, 0.10, 0.20)):
    """power 越低越好 → 取最小的 K 个"""
    n = pred.size(0)
    out = {}
    for r in k_ratios:
        k = max(1, int(round(n * r)))
        true_topk = set(torch.topk(true, k, largest=False).indices.cpu().tolist())
        pred_topk = set(torch.topk(pred, k, largest=False).indices.cpu().tolist())
        out[r] = len(true_topk & pred_topk) / k
    return out


def _metrics(pred, true):
    p = pred.cpu().numpy()
    t = true.cpu().numpy()
    tau, tau_p = stats.kendalltau(p, t)
    rho, _ = stats.spearmanr(p, t)
    pr, _ = stats.pearsonr(p, t)
    mape = np.mean(np.abs(t - p) / (np.abs(t) + 1e-8)) * 100
    mse = np.mean((p - t) ** 2)
    recalls = _topk_recall(pred, true)
    return {
        "n": len(p),
        "tau": 0.0 if np.isnan(tau) else tau,
        "tau_pval": tau_p,
        "rho": 0.0 if np.isnan(rho) else rho,
        "r2": (pr ** 2) if not np.isnan(pr) else 0.0,
        "mape": mape,
        "mse": mse,
        "recalls": recalls,
    }


def _build_model_from_ckpt(ckpt, device):
    """从 ckpt 元数据复原 ArithProxyGNN / PureGIN / OneHotGIN。"""
    model_class = ckpt.get("model_class", "")
    if model_class == "OneHotGIN" or ckpt.get("use_onehot_only", False):
        model = OneHotGIN(
            node_feature_dim=ckpt.get("node_feature_dim", 13),
            hidden_dim=ckpt.get("hidden_dim", 96),
            num_gnn_layers=ckpt.get("num_gnn_layers", 4),
            dropout=ckpt.get("dropout", 0.0),
            onehot_start=ckpt.get("onehot_start", 3),
            onehot_dim=ckpt.get("onehot_dim", 4),
        ).to(device)
    elif ckpt.get("use_pure_gcn", False):
        model = PureGCN(
            node_feature_dim=ckpt.get("node_feature_dim", 13),
            hidden_dim=ckpt.get("hidden_dim", 96),
            num_gnn_layers=ckpt.get("num_gnn_layers", 4),
        ).to(device)
    elif ckpt.get("use_pure_gin", False):
        model = PureGIN(
            node_feature_dim=ckpt.get("node_feature_dim", 13),
            hidden_dim=ckpt.get("hidden_dim", 96),
            num_gnn_layers=ckpt.get("num_gnn_layers", 4),
        ).to(device)
    else:
        model = ArithProxyGNN(
            node_feature_dim=ckpt.get("node_feature_dim", 13),
            hidden_dim=ckpt.get("hidden_dim", 96),
            num_gnn_layers=ckpt.get("num_gnn_layers", 4),
            dropout=ckpt.get("dropout", 0.15),
            use_mean_agg=ckpt.get("use_mean_agg", True),
            use_edge_feat=ckpt.get("use_edge_feat", True),
            external_edge_attr_dim=ckpt.get("external_edge_attr_dim", 0),
            use_typed_edges=ckpt.get("use_typed_edges", False),
            use_multitask=ckpt.get("use_multitask", False),
            use_jk_pool=ckpt.get("use_jk_pool", False),
            use_gin=ckpt.get("use_gin", False),
        ).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing or unexpected:
        print(f"  ⚠️ load_state_dict missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:    print(f"     missing[:3]: {missing[:3]}")
        if unexpected: print(f"     unexpected[:3]: {unexpected[:3]}")
    model.eval()
    return model


def _print_ckpt_config(ckpt):
    print(f"  📦 ckpt 配置:")
    print(f"     model_class       = {ckpt.get('model_class', 'unknown')}")
    print(f"     target            = {ckpt.get('target', 'power (legacy)')}")
    print(f"     node_feature_dim  = {ckpt.get('node_feature_dim', '?')}")
    print(f"     hidden_dim        = {ckpt.get('hidden_dim', '?')}")
    print(f"     num_gnn_layers    = {ckpt.get('num_gnn_layers', '?')}")
    print(f"     external_edge_attr_dim = {ckpt.get('external_edge_attr_dim', '?')}")
    print(f"     use_mean_agg      = {ckpt.get('use_mean_agg', '?')}")
    print(f"     use_edge_feat     = {ckpt.get('use_edge_feat', '?')}")
    print(f"     use_typed_edges   = {ckpt.get('use_typed_edges', False)}  (方案 3)")
    print(f"     use_multitask     = {ckpt.get('use_multitask', False)}  (方案 2)")
    print(f"     use_jk_pool       = {ckpt.get('use_jk_pool', False)}  (方案 4)")
    print(f"     use_gin           = {ckpt.get('use_gin', False)}  (GIN backbone)")
    if ckpt.get("model_class") == "OneHotGIN" or ckpt.get("use_onehot_only", False):
        print(f"     onehot slice      = X[:, {ckpt.get('onehot_start', 3)}:{ckpt.get('onehot_start', 3) + ckpt.get('onehot_dim', 4)}]")
    print(f"     train best_tau    = {ckpt.get('best_tau', '?')}")


def _select_val_subset(dataset, kfold, fold_id, val_ratio):
    """复现训练时的 val 划分.

    优先 kfold (与 train_proxy 的 KFold(seed=42) 对齐); 否则 random_split (val_ratio).
    """
    n = len(dataset)
    if kfold is not None:
        assert 0 <= fold_id < kfold, f"fold_id 必须在 [0, {kfold})"
        kf = KFold(n_splits=kfold, shuffle=True, random_state=42)
        splits = list(kf.split(range(n)))
        train_idx, val_idx = splits[fold_id]
        print(f"  📌 K-Fold {fold_id+1}/{kfold} 划分: train={len(train_idx)}, val={len(val_idx)}")
        return Subset(dataset, val_idx)

    val_size = max(int(val_ratio * n), 20)
    train_size = n - val_size
    _, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"  📌 random_split (val_ratio={val_ratio}): val={val_size}")
    return val_ds


def _run_inference(model, loader, device, power_mean, power_std):
    """跑一遍 val_loader, 返回 (pred_raw, true_raw, n_per_sample) 张量"""
    preds, trues, ns = [], [], []
    with torch.no_grad():
        for (X, edge_index, edge_attr, mask, _pn, pr,
             _npw, _nm, _ar, _dl) in loader:
            X = X.to(device, non_blocking=True)
            edge_index = edge_index.to(device, non_blocking=True)
            if edge_attr is not None:
                edge_attr = edge_attr.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            pred_norm = model(X, edge_index, mask, edge_attr=edge_attr)
            pred_raw = pred_norm * power_std + power_mean
            preds.append(pred_raw.cpu())
            trues.append(pr)
            ns.append(mask.sum(dim=1).long().cpu())
    return torch.cat(preds), torch.cat(trues), torch.cat(ns)


def _print_metrics(tag, m):
    r = m["recalls"]
    print(f"  [{tag}] n={m['n']:4d} | τ={m['tau']:+.4f} (p={m['tau_pval']:.1e}) "
          f"| ρ={m['rho']:+.4f} | R²={m['r2']:.4f}")
    print(f"           R@5%={r[0.05]:.3f}  R@10%={r[0.10]:.3f}  R@20%={r[0.20]:.3f}  "
          f"| MAPE={m['mape']:.2f}%  MSE={m['mse']:.2e}")


_TARGET_UNIT = {"power": "mW", "area": "um²", "delay": "ns"}


def _axis_label(target, kind):
    unit = _TARGET_UNIT.get(target, "")
    label = f"{kind} {target.capitalize()}"
    return f"{label} ({unit})" if unit else label


def _scatter_plot(preds, trues, m_mix, m_fix, save_path, title_extra="", target="power"):
    fig, axes = plt.subplots(1, 2 if m_fix is not None else 1,
                              figsize=(8 * (2 if m_fix is not None else 1), 7))
    if m_fix is None:
        axes = [axes]

    p_all = preds.numpy()
    t_all = trues.numpy()
    axes[0].scatter(t_all, p_all, alpha=0.5, c="dodgerblue", edgecolors="k", s=20)
    lo = min(t_all.min(), p_all.min())
    hi = max(t_all.max(), p_all.max())
    pad = (hi - lo) * 0.05
    axes[0].plot([lo - pad, hi + pad], [lo - pad, hi + pad], "r--", lw=1.5, label="y=x")
    axes[0].set_xlabel(_axis_label(target, "True"))
    axes[0].set_ylabel(_axis_label(target, "Predicted"))
    axes[0].set_title(
        f"Mixed-N (all val)\nn={m_mix['n']}  τ={m_mix['tau']:+.3f}  R²={m_mix['r2']:.3f}  "
        f"MAPE={m_mix['mape']:.1f}%"
    )
    axes[0].legend()
    axes[0].grid(True, ls=":", alpha=0.5)

    if m_fix is not None:
        # 第二张: 只画 fix-N 子集
        # m_fix 不携带 mask 信息, 但我们在调用前已经过滤了 — 传 preds/trues 全集这里就只能在外部画
        # → 调用方需要传入子集
        pass

    fig.suptitle(title_extra, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📈 scatter 保存到: {save_path}")


def evaluate_single_ckpt(ckpt_path, data_path, val_ratio=0.2,
                         kfold=None, fold_id=None, eval_filter_n=730,
                         batch_size=256, save_fig=True, fig_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 70}")
    print(f"  🖥  Device: {device}")
    print(f"  📂 ckpt: {ckpt_path}")
    print(f"  📂 data: {data_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    _print_ckpt_config(ckpt)

    # 1. 复原模型
    model = _build_model_from_ckpt(ckpt, device)

    # 2. 加载数据 — target 自动从 ckpt 读取 (老 ckpt 无该字段 fallback "power")
    target = ckpt.get("target", "power")
    full_ds = ArithDataset(data_path, target=target)
    pmean = float(ckpt.get("power_mean", full_ds.power_mean.item()))
    pstd = float(ckpt.get("power_std", full_ds.power_std.item()))
    print(f"  ⚖  使用 ckpt 的 power_mean={pmean:.6f}, power_std={pstd:.6f}")

    # 3. val 子集 (kfold 模式: 对齐训练; 否则 random_split)
    auto_fold = fold_id
    if kfold is not None and auto_fold is None:
        auto_fold = ckpt.get("fold_id", 0)
        print(f"  ℹ  从 ckpt 读 fold_id={auto_fold}")
    val_ds = _select_val_subset(full_ds, kfold, auto_fold, val_ratio)
    loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=custom_collate, num_workers=0, pin_memory=True,
    )

    # 4. 推理
    print(f"  🚀 推理 (batch_size={batch_size}) ...")
    preds, trues, ns = _run_inference(
        model, loader, device,
        torch.tensor(pmean, device=device), torch.tensor(pstd, device=device),
    )

    # 5. 指标 — mixed-N (全 val) + fixed-N (N==eval_filter_n)
    print()
    m_mix = _metrics(preds, trues)
    _print_metrics(f"mix (N 全集)", m_mix)

    m_fix = None
    if eval_filter_n is not None:
        keep = (ns == eval_filter_n)
        n_keep = int(keep.sum().item())
        if n_keep >= 16:
            m_fix = _metrics(preds[keep], trues[keep])
            _print_metrics(f"fix N={eval_filter_n}", m_fix)
        else:
            print(f"  ⚠️ fix-N={eval_filter_n} 仅 {n_keep} 样本, 跳过")

    # 6. 散点图
    if save_fig:
        fig_dir = fig_dir or "outputs"
        os.makedirs(fig_dir, exist_ok=True)
        tag = os.path.basename(ckpt_path).replace(".pth", "")
        fig_path = os.path.join(fig_dir, f"scatter_{tag}.png")
        # 画 mixed + (可选) fixed 两张子图
        if m_fix is not None:
            keep = (ns == eval_filter_n)
            fig, axes = plt.subplots(1, 2, figsize=(15, 7))
            for ax, p, t, title in [
                (axes[0], preds.numpy(), trues.numpy(),
                 f"Mixed-N\nn={m_mix['n']} τ={m_mix['tau']:+.3f} R²={m_mix['r2']:.3f} MAPE={m_mix['mape']:.1f}%"),
                (axes[1], preds[keep].numpy(), trues[keep].numpy(),
                 f"Fixed-N={eval_filter_n}\nn={m_fix['n']} τ={m_fix['tau']:+.3f} R²={m_fix['r2']:.3f} MAPE={m_fix['mape']:.1f}%"),
            ]:
                ax.scatter(t, p, alpha=0.5, c="dodgerblue", edgecolors="k", s=20)
                lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
                pad = (hi - lo) * 0.05
                ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "r--", lw=1.5)
                ax.set_xlabel(_axis_label(target, "True"))
                ax.set_ylabel(_axis_label(target, "Predicted"))
                ax.set_title(title)
                ax.grid(True, ls=":", alpha=0.5)
            fig.suptitle(tag, fontsize=11)
            plt.tight_layout()
            plt.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            _scatter_plot(preds, trues, m_mix, None, fig_path, title_extra=tag, target=target)
        print(f"  📈 scatter: {fig_path}")

    return {"mix": m_mix, "fix": m_fix, "fold_id": auto_fold}


def evaluate_kfold_glob(ckpt_glob, data_path, kfold=5,
                       eval_filter_n=730, batch_size=256, fig_dir=None):
    """评估一组 fold ckpt, 汇总 5-fold 平均"""
    ckpt_paths = sorted(glob.glob(ckpt_glob))
    if not ckpt_paths:
        print(f"❌ 没找到 ckpt 匹配 {ckpt_glob}")
        return

    print(f"\n📚 匹配到 {len(ckpt_paths)} 个 ckpt")
    for p in ckpt_paths:
        print(f"   - {p}")

    results = []
    for p in ckpt_paths:
        # 文件名 fold{N}.pth → fold_id (优先 ckpt 里的)
        res = evaluate_single_ckpt(
            p, data_path, kfold=kfold, fold_id=None,
            eval_filter_n=eval_filter_n, batch_size=batch_size,
            fig_dir=fig_dir,
        )
        results.append(res)

    # 汇总
    print(f"\n{'=' * 70}")
    print(f"  📊 K-Fold 汇总 ({len(results)} folds)")
    print(f"{'=' * 70}")
    for tag in ["mix", "fix"]:
        ms = [r[tag] for r in results if r[tag] is not None]
        if not ms:
            continue
        taus = [m["tau"] for m in ms]
        rhos = [m["rho"] for m in ms]
        recall10 = [m["recalls"][0.10] for m in ms]
        recall5  = [m["recalls"][0.05] for m in ms]
        mape = [m["mape"] for m in ms]
        print(f"\n  ▼ {tag}  (n={ms[0]['n']} 每 fold)")
        print(f"     τ      : {np.mean(taus):+.4f} ± {np.std(taus):.4f}  "
              f"per fold: {['{:+.3f}'.format(t) for t in taus]}")
        print(f"     ρ      : {np.mean(rhos):+.4f} ± {np.std(rhos):.4f}")
        print(f"     R@5%   : {np.mean(recall5):.3f} ± {np.std(recall5):.3f}")
        print(f"     R@10%  : {np.mean(recall10):.3f} ± {np.std(recall10):.3f}")
        print(f"     MAPE   : {np.mean(mape):.2f}% ± {np.std(mape):.2f}%")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None,
                        help="单个 ckpt 路径")
    parser.add_argument("--ckpt_glob", type=str, default=None,
                        help="glob 匹配多个 fold ckpt (汇总评估), 与 --ckpt 二选一")
    parser.add_argument("--data", type=str,
                        default="dataset/glitch_power_data_16bit_v2_7k_edge10.pt",
                        help="数据集 .pt 路径")
    parser.add_argument("--kfold", type=int, default=5,
                        help="K-Fold 切分数 (与训练时一致, seed=42 复现 val 子集)")
    parser.add_argument("--fold_id", type=int, default=None,
                        help="指定 fold_id; 默认从 ckpt 里读")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="kfold=None 时的 random_split val 比例")
    parser.add_argument("--no_kfold", action="store_true",
                        help="不用 KFold, 用 random_split(val_ratio)")
    parser.add_argument("--eval_filter_n", type=int, default=730,
                        help="fix-N 子集大小; 设 0 关闭 fix-N")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--fig_dir", type=str, default="outputs",
                        help="散点图保存目录")
    args = parser.parse_args()

    kfold_arg = None if args.no_kfold else args.kfold
    filter_n = None if args.eval_filter_n <= 0 else args.eval_filter_n

    if args.ckpt_glob is not None:
        evaluate_kfold_glob(
            args.ckpt_glob, args.data, kfold=kfold_arg,
            eval_filter_n=filter_n, batch_size=args.batch_size,
            fig_dir=args.fig_dir,
        )
    elif args.ckpt is not None:
        evaluate_single_ckpt(
            args.ckpt, args.data,
            val_ratio=args.val_ratio,
            kfold=kfold_arg, fold_id=args.fold_id,
            eval_filter_n=filter_n,
            batch_size=args.batch_size,
            fig_dir=args.fig_dir,
        )
    else:
        parser.error("--ckpt 或 --ckpt_glob 至少提供一个")
