"""
汇总 5 fold 的 fix-N=730 ranking 指标 (mean ± std)。

用法:
    python3 scripts/summarize_5fold.py --ckpt_prefix dataset/glitch_power_proxy_gnn_B
"""
import os
import sys
import argparse
import numpy as np
import torch
import scipy.stats as stats
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trainer"))
from train_proxy import ArithDataset, custom_collate
from diagnose_proxy import load_model, forward_with_intermediates


def evaluate_fold(model, dataset, fold_id, filter_n, device, batch_size=256):
    """在 fold {fold_id} 的 val 上跑 fix-N 指标"""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(kf.split(range(len(dataset))))
    _, val_idx = splits[fold_id]

    # filter to fix-N
    sample_ns = np.array([dataset.data[i]["X"].shape[0] for i in val_idx])
    val_idx_fix = val_idx[sample_ns == filter_n]
    if len(val_idx_fix) < 16:
        print(f"  Fold {fold_id}: fix-N={filter_n} 样本太少 ({len(val_idx_fix)}), 跳过")
        return None

    loader = DataLoader(
        Subset(dataset, val_idx_fix), batch_size=batch_size,
        shuffle=False, collate_fn=custom_collate, num_workers=0,
    )
    preds, trues = [], []
    for X, ei, mask, pn, pr in loader:
        X = X.to(device); ei = ei.to(device); mask = mask.to(device)
        out = forward_with_intermediates(model, X, ei, mask)
        preds.append(out["pred"].cpu())
        trues.append(pn)
    preds = torch.cat(preds); trues = torch.cat(trues)
    n = preds.size(0)

    p_np = preds.numpy(); t_np = trues.numpy()
    tau = stats.kendalltau(p_np, t_np).correlation
    rho = stats.spearmanr(p_np, t_np).correlation
    pear = float(np.corrcoef(p_np, t_np)[0, 1])

    recalls = {}
    for r in (0.05, 0.10, 0.20):
        k = max(1, int(round(n * r)))
        ti = set(torch.topk(trues, k, largest=False).indices.tolist())
        pi = set(torch.topk(preds, k, largest=False).indices.tolist())
        recalls[r] = len(ti & pi) / k

    return {
        "n": n, "pearson": pear, "tau": tau, "rho": rho,
        "R@5%": recalls[0.05], "R@10%": recalls[0.10], "R@20%": recalls[0.20],
    }


def main(ckpt_prefix, data_path, filter_n):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  🖥  Device: {device}")
    print(f"  📂 数据集: {data_path}")
    dataset = ArithDataset(data_path)

    print(f"\n  {'='*78}")
    print(f"  5-Fold 汇总: ckpt_prefix={ckpt_prefix}, fix-N={filter_n}")
    print(f"  {'='*78}\n")

    fold_metrics = []
    for fold_id in range(5):
        ckpt_path = f"{ckpt_prefix}_fold{fold_id}.pth"
        if not os.path.exists(ckpt_path):
            print(f"  ⏭  Fold {fold_id}: ckpt 不存在 ({ckpt_path}), 跳过")
            continue
        model, _ = load_model(ckpt_path, device)
        m = evaluate_fold(model, dataset, fold_id, filter_n, device)
        if m is not None:
            fold_metrics.append((fold_id, m))
            print(f"  Fold {fold_id} (n={m['n']}): "
                  f"τ={m['tau']:+.4f}  ρ={m['rho']:+.4f}  "
                  f"R@5={m['R@5%']:.3f}  R@10={m['R@10%']:.3f}  R@20={m['R@20%']:.3f}  "
                  f"Pearson={m['pearson']:+.4f}")
        del model
        torch.cuda.empty_cache()

    if not fold_metrics:
        print("\n  ❌ 没有可用的 fold ckpt")
        return

    print(f"\n  {'='*78}")
    print(f"  📊 跨 fold 汇总 (n_folds={len(fold_metrics)})")
    print(f"  {'='*78}")
    keys = ["tau", "rho", "R@5%", "R@10%", "R@20%", "pearson"]
    for k in keys:
        vals = [m[k] for _, m in fold_metrics]
        print(f"  {k:>10s}: {np.mean(vals):+.4f} ± {np.std(vals):.4f}  "
              f"[min={min(vals):+.4f}, max={max(vals):+.4f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_prefix", default="dataset/glitch_power_proxy_gnn_B",
                        help="ckpt 路径前缀 (脚本会找 <prefix>_fold{0,1,2,3,4}.pth)")
    parser.add_argument("--data", default="dataset/glitch_power_data_16bit_enriched.pt")
    parser.add_argument("--filter_n", type=int, default=730)
    args = parser.parse_args()
    main(args.ckpt_prefix, args.data, args.filter_n)
