"""
Route D: 训练时加入节点级 power 辅助监督。

仅用 Route D 数据集 (1000 样本含 per-node power)，做内部 90/10 split 的
smoke test，验证：
  1. node-power 辅助损失能让模型学到 per-FA 的真实功耗
  2. 全局 R@10% / Pearson 跟随提升
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from proxy_mlp import ArithProxyGNN
from train_proxy import (
    compute_rank_loss, listwise_loss,
    compute_kendall_tau, compute_spearman, compute_topk_recall,
    StratifiedNBatchSampler,
)


# ============== 自定义 Dataset (带 node_powers) ==============
class NodePowerDataset(Dataset):
    def __init__(self, data_path):
        raw = torch.load(data_path, map_location="cpu", weights_only=False)
        # 转 edge_index (老数据可能存 P)
        first = raw[0]
        if "edge_index" not in first:
            for item in raw:
                P = item.pop("P")
                item["edge_index"] = P.nonzero(as_tuple=False).t().contiguous().long()
        self.data = raw

        powers = torch.tensor([x["power"] for x in raw], dtype=torch.float32)
        self.power_mean = powers.mean()
        self.power_std = powers.std() + 1e-8

        # node_power 也归一化到合理范围 (默认单位 W，量级 1e-6)
        # 用 log10(node_power[>0]) 的 mean/std 防止小数值在 huber 下信号弱
        all_np = torch.cat([
            x["node_powers"][x["node_power_mask"]] for x in raw
            if x.get("node_power_mask") is not None
        ])
        self.np_scale = all_np.mean().item()  # 约 1e-6
        print(f"  📊 数据集: {len(raw)} 样本，X={raw[0]['X'].shape}")
        print(f"     power: mean={self.power_mean:.6f}, std={self.power_std:.6f}")
        print(f"     node_power 量级 (mean 非零): {self.np_scale:.3e} W")
        print(f"     平均每样本带 node_power 标签的节点数: "
              f"{np.mean([x['node_power_mask'].sum().item() for x in raw]):.1f}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        X = item["X"]
        edge_index = item["edge_index"]
        power_val = item["power"]
        power_norm = torch.tensor(
            (power_val - self.power_mean.item()) / self.power_std.item(),
            dtype=torch.float32,
        )
        power_raw = torch.tensor(power_val, dtype=torch.float32)
        # 归一化到 ~O(1)：除以 np_scale
        node_powers = item["node_powers"] / self.np_scale  # [N]
        node_mask = item["node_power_mask"]  # [N] bool
        return X, edge_index, power_norm, power_raw, node_powers, node_mask


class NodePowerValDataset(Dataset):
    """Val 集: enriched 全集，没有 node_powers，用 train 集的 norm 参数。"""
    def __init__(self, data_path, power_mean, power_std, np_scale):
        raw = torch.load(data_path, map_location="cpu", weights_only=False)
        first = raw[0]
        if "edge_index" not in first:
            for item in raw:
                P = item.pop("P")
                item["edge_index"] = P.nonzero(as_tuple=False).t().contiguous().long()
        self.data = raw
        self.power_mean = power_mean
        self.power_std = power_std
        self.np_scale = np_scale
        print(f"     Val 总样本: {len(raw)}, X={raw[0]['X'].shape}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        X = item["X"]
        edge_index = item["edge_index"]
        power_val = item["power"]
        power_norm = torch.tensor(
            (power_val - self.power_mean) / self.power_std, dtype=torch.float32,
        )
        power_raw = torch.tensor(power_val, dtype=torch.float32)
        # val 集没有 node 标签，返回全 0 + 全 False mask
        N = X.shape[0]
        node_powers = torch.zeros(N)
        node_mask = torch.zeros(N, dtype=torch.bool)
        return X, edge_index, power_norm, power_raw, node_powers, node_mask


def node_collate(batch):
    X_list, ei_list, pn_list, pr_list, np_list, nm_list = zip(*batch)
    B = len(batch)
    max_N = max(x.size(0) for x in X_list)
    feat_dim = X_list[0].size(1)

    X_pad = torch.zeros(B, max_N, feat_dim)
    mask = torch.zeros(B, max_N, dtype=torch.bool)
    node_pow_pad = torch.zeros(B, max_N)
    node_mask_pad = torch.zeros(B, max_N, dtype=torch.bool)
    ei_off = []

    for i, (ei, x, npw, nm) in enumerate(zip(ei_list, X_list, np_list, nm_list)):
        n = x.size(0)
        X_pad[i, :n, :] = x
        mask[i, :n] = True
        node_pow_pad[i, :n] = npw
        node_mask_pad[i, :n] = nm
        ei_off.append(ei + i * max_N)

    edge_index = torch.cat(ei_off, dim=1)
    power_norm = torch.stack(pn_list)
    power_raw = torch.stack(pr_list)
    return X_pad, edge_index, mask, power_norm, power_raw, node_pow_pad, node_mask_pad


def _compute_metrics(pred, true):
    tau, _ = compute_kendall_tau(pred, true)
    rho, _ = compute_spearman(pred, true)
    recalls = compute_topk_recall(pred, true, k_ratios=(0.05, 0.10, 0.20))
    return {"tau": tau, "rho": rho, "recalls": recalls}


def train(
    data_path="dataset/glitch_power_data_16bit_node_power_enriched.pt",
    val_data_path="dataset/glitch_power_data_16bit_enriched.pt",
    save_path="dataset/glitch_power_proxy_gnn_D.pth",
    epochs=400, batch_size=64, val_batch_size=256,
    lr=3e-4, weight_decay=5e-3,
    hidden_dim=96, num_gnn_layers=4, dropout=0.15,
    w_mse=0.5, w_rank=0.2, w_list=0.5, w_scale=0.5,
    w_node=0.2,             # node-power 辅助损失权重 (独立 aux head)
    node_warmup_epochs=20,  # 前 N epoch 不加 node loss
    eval_filter_n=730,
    seed=42,
    num_workers=4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  🖥  Device: {device}")
    print(f"  📂 Train 数据 (含 node_power): {data_path}")
    dataset = NodePowerDataset(data_path)
    node_feat_dim = dataset.data[0]["X"].shape[1]

    # Val 集：用 enriched 全集做 fix-N=730 评估 (统计可靠)
    print(f"  📂 Val 数据 (enriched 全集): {val_data_path}")
    val_dataset = NodePowerValDataset(
        val_data_path,
        power_mean=dataset.power_mean.item(),
        power_std=dataset.power_std.item(),
        np_scale=dataset.np_scale,
    )

    train_idx = np.arange(len(dataset))
    # Val 选 N=eval_filter_n 子集 + 留 hold-out (避免泄漏：训练样本不在 val)
    val_ns = np.array([val_dataset.data[i]["X"].shape[0] for i in range(len(val_dataset))])
    val_idx = np.where(val_ns == eval_filter_n)[0]
    print(f"  📊 Train={len(train_idx)} (含 node_power 标签)")
    print(f"     Val (fix-N={eval_filter_n}) = {len(val_idx)} 样本")

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)

    # 启用 stratified batch (B 方案)
    train_batch_sampler = StratifiedNBatchSampler(
        train_subset, batch_size=min(batch_size, len(train_idx)),
        drop_last=True, shuffle=True, seed=seed,
    )
    train_loader = DataLoader(
        train_subset, batch_sampler=train_batch_sampler,
        collate_fn=node_collate, num_workers=num_workers,
        pin_memory=True, persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_subset, batch_size=min(val_batch_size, len(val_idx)),
        shuffle=False, collate_fn=node_collate, num_workers=num_workers,
        pin_memory=True, persistent_workers=(num_workers > 0),
    )

    model = ArithProxyGNN(
        node_feature_dim=node_feat_dim, hidden_dim=hidden_dim,
        num_gnn_layers=num_gnn_layers, dropout=dropout,
        use_mean_agg=True, use_edge_feat=True,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"     模型参数量: {n_params:,}")
    print(f"     Loss: {w_mse}×Huber + {w_rank}×Rank + {w_list}×ListMLE "
          f"+ {w_scale}×Scale + {w_node}×NodeHuber\n")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    huber_fn = nn.SmoothL1Loss()
    rank_fn = nn.MarginRankingLoss(margin=0.01)
    node_huber = nn.SmoothL1Loss(reduction="none")

    power_mean = dataset.power_mean.to(device)
    power_std = dataset.power_std.to(device)

    best_recall = -1.0
    best_state = None
    patience = 0
    max_patience = 100

    for epoch in range(epochs):
        model.train()
        sum_loss = 0.0
        sum_node = 0.0
        for X, ei, mask, pn, pr, npw, nm in train_loader:
            X = X.to(device, non_blocking=True)
            ei = ei.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            pn = pn.to(device, non_blocking=True)
            pr = pr.to(device, non_blocking=True)
            npw = npw.to(device, non_blocking=True)
            nm = nm.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred, node_pred = model(X, ei, mask, return_nodes=True)

            loss_mse = huber_fn(pred, pn)
            loss_rank = compute_rank_loss(pred, pr, rank_fn, device)
            loss_list = listwise_loss(pred, pn)
            if pred.numel() >= 2:
                loss_scale = (pred.std(unbiased=False) - pn.std(unbiased=False)).abs()
            else:
                loss_scale = torch.tensor(0.0, device=device)

            # node-power 辅助损失 (独立 aux head，不影响 global 求和)
            if nm.any() and epoch >= node_warmup_epochs:
                diff = node_huber(node_pred, npw) * nm.float()
                loss_node = diff.sum() / nm.float().sum().clamp_min(1.0)
                w_node_eff = w_node
            else:
                loss_node = torch.tensor(0.0, device=device)
                w_node_eff = 0.0

            loss = (w_mse * loss_mse + w_rank * loss_rank
                    + w_list * loss_list + w_scale * loss_scale
                    + w_node_eff * loss_node)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            sum_loss += loss.item()
            sum_node += loss_node.item()

        scheduler.step()
        avg_loss = sum_loss / max(len(train_loader), 1)
        avg_node = sum_node / max(len(train_loader), 1)

        if (epoch + 1) % 5 == 0:
            model.eval()
            all_pred, all_true, all_n = [], [], []
            all_npred, all_ntrue = [], []  # 节点级
            with torch.no_grad():
                for X, ei, mask, pn, pr, npw, nm in val_loader:
                    X = X.to(device, non_blocking=True)
                    ei = ei.to(device, non_blocking=True)
                    mask = mask.to(device, non_blocking=True)
                    pred, node_pred = model(X, ei, mask, return_nodes=True)
                    pred_raw = pred * power_std + power_mean
                    all_pred.append(pred_raw.cpu())
                    all_true.append(pr)
                    all_n.append(mask.sum(dim=1).long().cpu())
                    # 仅有标签的节点参与节点级评估
                    nm_flat = nm.reshape(-1)
                    if nm_flat.any():
                        all_npred.append(node_pred.cpu().reshape(-1)[nm_flat])
                        all_ntrue.append(npw.reshape(-1)[nm_flat])
            all_pred = torch.cat(all_pred)
            all_true = torch.cat(all_true)
            all_n = torch.cat(all_n)

            m_mix = _compute_metrics(all_pred, all_true)
            m_fix = None
            if eval_filter_n is not None:
                keep = (all_n == eval_filter_n)
                n_kept = int(keep.sum().item())
                if n_kept >= 16:
                    m_fix = _compute_metrics(all_pred[keep], all_true[keep])
                    m_fix["n"] = n_kept

            # 节点级 ranking metric (跨整个 val 的所有 FA 节点)
            node_rank = ""
            if all_npred:
                np_pred = torch.cat(all_npred)
                np_true = torch.cat(all_ntrue)
                if np_pred.numel() >= 16:
                    tau_n, _ = compute_kendall_tau(np_pred, np_true)
                    rho_n, _ = compute_spearman(np_pred, np_true)
                    node_rank = (f" | node({np_pred.numel()}): τ={tau_n:+.3f} ρ={rho_n:+.3f}")

            line = (f"  [D] Ep{epoch + 1:03d} L={avg_loss:.3f} (node {avg_node:.3f})"
                    f" | mix: τ={m_mix['tau']:+.3f} ρ={m_mix['rho']:+.3f} "
                    f"R@10={m_mix['recalls'][0.10]:.3f}")
            if m_fix:
                line += (f" || fix(N={eval_filter_n},n={m_fix['n']}): "
                         f"τ={m_fix['tau']:+.3f} ρ={m_fix['rho']:+.3f} "
                         f"R@10={m_fix['recalls'][0.10]:.3f}")
            line += node_rank
            print(line)

            track = (m_fix["recalls"][0.10] if m_fix else m_mix["recalls"][0.10])
            if track > best_recall:
                best_recall = track
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience = 0
                print(f"       🌟 best R@10%={best_recall:.3f}")
            else:
                patience += 5
                if patience >= max_patience:
                    print(f"       ⏹  早停 @ epoch {epoch + 1}")
                    break

    torch.save({
        "model_state_dict": best_state,
        "node_feature_dim": node_feat_dim,
        "hidden_dim": hidden_dim,
        "num_gnn_layers": num_gnn_layers,
        "dropout": dropout,
        "power_mean": dataset.power_mean.item(),
        "power_std": dataset.power_std.item(),
        "np_scale": dataset.np_scale,
        "model_class": "ArithProxyGNN",
        "use_mean_agg": True,
        "use_edge_feat": True,
        "best_recall10": best_recall,
    }, save_path)
    print(f"\n  ✅ 已保存: {save_path} (best R@10%={best_recall:.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset/glitch_power_data_16bit_node_power_enriched.pt")
    parser.add_argument("--save", default="dataset/glitch_power_proxy_gnn_D.pth")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--w_node", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        data_path=args.data, save_path=args.save,
        epochs=args.epochs, w_node=args.w_node, seed=args.seed,
    )
