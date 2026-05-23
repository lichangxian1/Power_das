import os
import torch
import torch.nn as nn
import numpy as np
import scipy.stats as stats
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import KFold

from proxy_mlp import ArithProxyGNN     # ← 新 GNN


# ========================== 数据集 ==========================
class ArithDataset(Dataset):
    """
    自动兼容三种数据格式：
      - 旧格式 A: X[N, 7] + P[N, N]   (最原始)
      - 旧格式 B: X[N, 9] + P[N, N]   (augment_features 之后)
      - 新格式:   X[N, 9] + edge_index[2, E]
    旧格式加载时自动转换为新格式 (只在内存生效，不写盘)。
    """

    def __init__(self, data_path):
        raw = torch.load(data_path, map_location="cpu")

        first = raw[0]
        if "edge_index" not in first:
            if "P" not in first:
                raise ValueError("数据集既没有 edge_index 也没有 P 矩阵")
            print(f"  🔄 检测到旧格式 (dense P)，正在转换为稀疏 edge_index ...")
            for item in tqdm(raw, desc="P → edge_index"):
                P = item.pop("P")
                item["edge_index"] = P.nonzero(as_tuple=False).t().contiguous().long()
            print(f"  ✅ 转换完成 (仅在内存中生效)")

        self.data = raw

        powers = torch.tensor([item["power"] for item in raw], dtype=torch.float32)
        self.power_mean = powers.mean()
        self.power_std = powers.std() + 1e-8

        feat_dim = raw[0]["X"].shape[1]
        n_edges_avg = sum(item["edge_index"].shape[1] for item in raw) / len(raw)
        print(f"  📊 数据集: {len(raw)} 样本，X 维度={feat_dim}")
        print(f"     平均边数: {n_edges_avg:.0f}")
        print(f"     power: mean={self.power_mean:.6f}, std={self.power_std:.6f}, "
              f"min={powers.min():.6f}, max={powers.max():.6f}")

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
        return X, edge_index, power_norm, power_raw


# ========================== Collate ==========================
def custom_collate(batch):
    X_list, ei_list, pn_list, pr_list = zip(*batch)
    B = len(batch)
    max_N = max(x.size(0) for x in X_list)
    feat_dim = X_list[0].size(1)

    X_pad = torch.zeros(B, max_N, feat_dim)
    mask = torch.zeros(B, max_N, dtype=torch.bool)
    ei_offset_list = []

    for i, (ei, x) in enumerate(zip(ei_list, X_list)):
        n = x.size(0)
        X_pad[i, :n, :] = x
        mask[i, :n] = True
        ei_offset_list.append(ei + i * max_N)

    edge_index = torch.cat(ei_offset_list, dim=1)

    power_norm = torch.stack(pn_list)
    power_raw = torch.stack(pr_list)
    return X_pad, edge_index, mask, power_norm, power_raw


# ========================== 工具 ==========================
def compute_kendall_tau(pred, true):
    p = pred.detach().cpu().numpy()
    t = true.detach().cpu().numpy()
    tau, pval = stats.kendalltau(p, t)
    return (tau if not np.isnan(tau) else 0.0), pval


def compute_rank_loss(pred, true, criterion, device):
    B = pred.size(0)
    if B < 2:
        return torch.tensor(0.0, device=device)

    idx_i, idx_j = torch.triu_indices(B, B, offset=1, device=device)

    max_pairs = 2048
    num_pairs = idx_i.size(0)
    if num_pairs > max_pairs:
        perm = torch.randperm(num_pairs, device=device)[:max_pairs]
        idx_i = idx_i[perm]
        idx_j = idx_j[perm]

    diff = true[idx_i] - true[idx_j]
    valid = (diff.abs() > 1e-8)

    if valid.sum() == 0:
        return torch.tensor(0.0, device=device)

    target = torch.sign(diff[valid])
    return criterion(pred[idx_i[valid]], pred[idx_j[valid]], target)


def listwise_loss(pred, true):
    _, sorted_indices = true.sort(descending=True)
    pred_sorted = pred[sorted_indices]

    max_val = pred_sorted.max()
    pred_shifted = pred_sorted - max_val
    cumsums = torch.logcumsumexp(pred_shifted.flip(0), dim=0).flip(0)
    loss = (cumsums - pred_shifted).mean()
    return loss


# ========================== 单 Fold 训练 ==========================
def train_one_fold(
    model, train_loader, val_loader,
    device, epochs, lr, weight_decay,
    w_mse, w_rank, w_list,
    power_mean, power_std, fold_id=0,
):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    huber_fn = nn.SmoothL1Loss()
    rank_fn = nn.MarginRankingLoss(margin=0.01)

    best_tau = -1.0
    best_state = None
    patience_counter = 0
    max_patience = 150

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for X, edge_index, mask, power_norm, power_raw in train_loader:
            X = X.to(device, non_blocking=True)
            edge_index = edge_index.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            power_norm = power_norm.to(device, non_blocking=True)
            power_raw = power_raw.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred = model(X, edge_index, mask)

            loss_mse = huber_fn(pred, power_norm)
            loss_rank = compute_rank_loss(pred, power_raw, rank_fn, device)
            loss_list = listwise_loss(pred, power_norm)

            loss = w_mse * loss_mse + w_rank * loss_rank + w_list * loss_list

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / max(len(train_loader), 1)

        if (epoch + 1) % 5 == 0:
            model.eval()
            all_pred, all_true = [], []

            with torch.no_grad():
                for X, edge_index, mask, pn, pr in val_loader:
                    X = X.to(device, non_blocking=True)
                    edge_index = edge_index.to(device, non_blocking=True)
                    mask = mask.to(device, non_blocking=True)
                    pred = model(X, edge_index, mask)
                    pred_raw = pred * power_std + power_mean
                    all_pred.append(pred_raw)
                    all_true.append(pr.to(device, non_blocking=True))

            all_pred = torch.cat(all_pred)
            all_true = torch.cat(all_true)

            tau, pval = compute_kendall_tau(all_pred, all_true)
            mse = nn.functional.mse_loss(all_pred, all_true).item()
            mape = (((all_pred - all_true).abs() / (all_true.abs() + 1e-8)).mean() * 100).item()

            print(f"  [Fold {fold_id}] Epoch {epoch + 1:03d} | Loss: {avg_loss:.4f} "
                  f"| τ={tau:+.4f} (p={pval:.4f}) | MSE={mse:.6f} | MAPE={mape:.2f}%")

            if tau > best_tau:
                best_tau = tau
                patience_counter = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                print(f"       🌟 Fold {fold_id} best τ = {best_tau:+.4f}")
            else:
                patience_counter += 5
                if patience_counter >= max_patience:
                    print(f"       ⏹  Fold {fold_id} 早停")
                    break

    return best_tau, best_state


# ========================== 主函数: K-Fold ==========================
def train_proxy(
    data_path: str = "dataset/glitch_power_data_16bit_merged.pt",
    save_path: str = "dataset/glitch_power_proxy_gnn.pth",
    n_folds: int = 5,
    batch_size: int = 64,
    val_batch_size: int = 256,
    epochs: int = 500,
    lr: float = 3e-4,             # ← GNN 参数多，LR 调小一些
    weight_decay: float = 5e-3,   # ← 加大 weight decay 防过拟合
    node_feature_dim: int = 9,
    hidden_dim: int = 64,
    num_gnn_layers: int = 3,      # ← 3 层 GNN
    dropout: float = 0.15,        # ← 略大于之前的 0.1
    w_mse: float = 1.0,
    w_rank: float = 0.2,
    w_list: float = 0.5,
    num_workers: int = 4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  🖥  Device: {device}")

    dataset = ArithDataset(data_path)
    power_mean = dataset.power_mean.to(device)
    power_std = dataset.power_std.to(device)

    # 自动对齐特征维度
    actual_dim = dataset.data[0]["X"].shape[1]
    if actual_dim != node_feature_dim:
        print(f"  ℹ  自动将 node_feature_dim 从 {node_feature_dim} 调整为 {actual_dim}")
        node_feature_dim = actual_dim

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_taus = []
    best_overall_tau = -1.0
    best_overall_state = None

    # 参数量统计
    tmp_model = ArithProxyGNN(node_feature_dim, hidden_dim, num_gnn_layers, dropout=dropout)
    n_params = sum(p.numel() for p in tmp_model.parameters())
    del tmp_model

    print(f"\n  🚀 开始 {n_folds}-Fold 交叉验证训练")
    print(f"     模型: ArithProxyGNN ({num_gnn_layers}-layer Bidirectional GNN")
    print(f"            + 异构投影 + 图级双池化)")
    print(f"     参数量: {n_params:,}")
    print(f"     hidden_dim={hidden_dim}, in_dim={node_feature_dim}, dropout={dropout}")
    print(f"     LR={lr}, weight_decay={weight_decay}")
    print(f"     Loss: {w_mse}×Huber + {w_rank}×RankLoss + {w_list}×ListMLE")
    print(f"     DataLoader: train_bs={batch_size}, val_bs={val_batch_size}, num_workers={num_workers}\n")

    for fold_id, (train_idx, val_idx) in enumerate(kf.split(range(len(dataset)))):
        print(f"  {'=' * 60}")
        print(f"  Fold {fold_id}: train={len(train_idx)}, val={len(val_idx)}")
        print(f"  {'=' * 60}")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        actual_bs = min(batch_size, len(train_idx))
        actual_val_bs = min(val_batch_size, len(val_idx))

        train_loader = DataLoader(
            train_subset,
            batch_size=actual_bs,
            shuffle=True,
            drop_last=True,
            collate_fn=custom_collate,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=actual_val_bs,
            shuffle=False,
            collate_fn=custom_collate,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )

        model = ArithProxyGNN(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_gnn_layers=num_gnn_layers,
            dropout=dropout,
        ).to(device)

        tau, state = train_one_fold(
            model, train_loader, val_loader,
            device, epochs, lr, weight_decay,
            w_mse, w_rank, w_list,
            power_mean, power_std, fold_id,
        )

        fold_taus.append(tau)
        if tau > best_overall_tau:
            best_overall_tau = tau
            best_overall_state = state

    # 汇总
    print(f"\n  {'=' * 60}")
    print(f"  📊 K-Fold 结果:")
    for i, t in enumerate(fold_taus):
        print(f"     Fold {i}: τ = {t:+.4f}")
    print(f"     平均 τ = {np.mean(fold_taus):+.4f} ± {np.std(fold_taus):.4f}")
    print(f"     最佳 τ = {best_overall_tau:+.4f}")
    print(f"  {'=' * 60}")

    torch.save({
        "model_state_dict": best_overall_state,
        "node_feature_dim": node_feature_dim,
        "hidden_dim": hidden_dim,
        "num_gnn_layers": num_gnn_layers,
        "dropout": dropout,
        "power_mean": dataset.power_mean.item(),
        "power_std": dataset.power_std.item(),
        "model_class": "ArithProxyGNN",
    }, save_path)
    print(f"\n  ✅ 最佳模型已保存至: {save_path}")


if __name__ == "__main__":
    train_proxy()