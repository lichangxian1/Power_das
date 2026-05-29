import os
import random
import torch
import torch.nn as nn
import numpy as np
import scipy.stats as stats
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset, Sampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import KFold

from proxy_mlp import ArithProxyGNN     # ← 新 GNN


# ========================== Stratified-N Batch Sampler ==========================
class StratifiedNBatchSampler(Sampler):
    """Batch 内所有样本节点数 N 相同 (按 X.shape[0] 分桶)。

    目的: 让 rank/list/scale loss 不被跨结构家族 (N 异构) 的样本污染,
          所有 batch 内的 pair 都在公平的"同结构"前提下比较。
    """

    def __init__(self, dataset_subset, batch_size, drop_last=True, shuffle=True, seed=42):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed

        # Subset: local_index → base_index, 通过 base.data[base_index]["X"].shape[0] 拿 N
        if isinstance(dataset_subset, Subset):
            base = dataset_subset.dataset
            base_indices = dataset_subset.indices
        else:
            base = dataset_subset
            base_indices = list(range(len(dataset_subset)))

        # 按 N 分桶 (存 local index, 即 dataset_subset 的索引)
        buckets = {}
        for local_i, base_i in enumerate(base_indices):
            n = base.data[int(base_i)]["X"].shape[0]
            buckets.setdefault(n, []).append(local_i)
        self.buckets = buckets
        self._epoch = 0

        # 信息日志 (第一次构造时)
        sizes = sorted(((n, len(ids)) for n, ids in buckets.items()), key=lambda x: -x[1])
        bucket_summary = ", ".join(f"N={n}:{c}" for n, c in sizes[:5])
        print(f"  [StratifiedNBatchSampler] {len(buckets)} 个 N 桶, "
              f"top5: {bucket_summary}{'...' if len(sizes) > 5 else ''}")
        print(f"     batch_size={batch_size}, drop_last={drop_last}, total_batches={len(self)}")

    def set_epoch(self, epoch):
        self._epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        all_batches = []
        for n, ids in self.buckets.items():
            ids = ids.copy()
            if self.shuffle:
                rng.shuffle(ids)
            for i in range(0, len(ids), self.batch_size):
                batch = ids[i:i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                all_batches.append(batch)
        if self.shuffle:
            rng.shuffle(all_batches)
        return iter(all_batches)

    def __len__(self):
        total = 0
        for ids in self.buckets.values():
            if self.drop_last:
                total += len(ids) // self.batch_size
            else:
                total += (len(ids) + self.batch_size - 1) // self.batch_size
        return total


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
        # v2 数据带 edge_attr; 旧数据没有 → 返回 None 占位
        edge_attr = item.get("edge_attr", None)
        power_val = item["power"]
        power_norm = torch.tensor(
            (power_val - self.power_mean.item()) / self.power_std.item(),
            dtype=torch.float32,
        )
        power_raw = torch.tensor(power_val, dtype=torch.float32)
        # node_power 标签 (Route D 用; 没有则填占位)
        node_powers = item.get("node_powers", None)
        node_mask = item.get("node_power_mask", None)
        if node_powers is None:
            N = X.shape[0]
            node_powers = torch.zeros(N)
            node_mask = torch.zeros(N, dtype=torch.bool)
        return X, edge_index, edge_attr, power_norm, power_raw, node_powers, node_mask


# ========================== Collate ==========================
def custom_collate(batch):
    X_list, ei_list, ea_list, pn_list, pr_list, np_list, nm_list = zip(*batch)
    B = len(batch)
    max_N = max(x.size(0) for x in X_list)
    feat_dim = X_list[0].size(1)

    X_pad = torch.zeros(B, max_N, feat_dim)
    mask = torch.zeros(B, max_N, dtype=torch.bool)
    npw_pad = torch.zeros(B, max_N)
    nmask_pad = torch.zeros(B, max_N, dtype=torch.bool)
    ei_offset_list = []

    for i, (ei, x, npw, nm) in enumerate(zip(ei_list, X_list, np_list, nm_list)):
        n = x.size(0)
        X_pad[i, :n, :] = x
        mask[i, :n] = True
        npw_pad[i, :n] = npw
        nmask_pad[i, :n] = nm
        ei_offset_list.append(ei + i * max_N)

    edge_index = torch.cat(ei_offset_list, dim=1)
    # edge_attr: 若所有样本都有则拼接, 否则返回 None
    if all(ea is not None for ea in ea_list):
        edge_attr = torch.cat(ea_list, dim=0)
    else:
        edge_attr = None

    power_norm = torch.stack(pn_list)
    power_raw = torch.stack(pr_list)
    return X_pad, edge_index, edge_attr, mask, power_norm, power_raw, npw_pad, nmask_pad


# ========================== 工具 ==========================
def compute_kendall_tau(pred, true):
    p = pred.detach().cpu().numpy()
    t = true.detach().cpu().numpy()
    tau, pval = stats.kendalltau(p, t)
    return (tau if not np.isnan(tau) else 0.0), pval


def compute_spearman(pred, true):
    p = pred.detach().cpu().numpy()
    t = true.detach().cpu().numpy()
    rho, pval = stats.spearmanr(p, t)
    return (rho if not np.isnan(rho) else 0.0), pval


def compute_topk_recall(pred, true, k_ratios=(0.05, 0.10, 0.20)):
    """
    Top-K Recall：真实 power 最低的 K 个样本中，模型预测的 Top-K 命中了多少。
    power 越低越好 → 取最小的 K 个 (largest=False)。
    返回 dict: {ratio: recall}
    """
    n = pred.size(0)
    results = {}
    for r in k_ratios:
        k = max(1, int(round(n * r)))
        true_topk = set(torch.topk(true, k, largest=False).indices.cpu().tolist())
        pred_topk = set(torch.topk(pred, k, largest=False).indices.cpu().tolist())
        hit = len(true_topk & pred_topk)
        results[r] = hit / k
    return results


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
def _compute_metrics(pred, true):
    """统一计算 τ, ρ, R@K, MSE, MAPE。"""
    tau, _ = compute_kendall_tau(pred, true)
    rho, _ = compute_spearman(pred, true)
    recalls = compute_topk_recall(pred, true, k_ratios=(0.05, 0.10, 0.20))
    mse = nn.functional.mse_loss(pred, true).item()
    mape = (((pred - true).abs() / (true.abs() + 1e-8)).mean() * 100).item()
    return {"tau": tau, "rho": rho, "recalls": recalls, "mse": mse, "mape": mape}


def train_one_fold(
    model, train_loader, val_loader,
    device, epochs, lr, weight_decay,
    w_mse, w_rank, w_list, w_scale,
    power_mean, power_std, fold_id=0,
    eval_filter_n=None,
    w_node=0.0, node_warmup_epochs=20,
):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    huber_fn = nn.SmoothL1Loss()
    rank_fn = nn.MarginRankingLoss(margin=0.01)
    node_huber = nn.SmoothL1Loss(reduction="none")

    best_tau = -1.0
    best_spearman = -1.0
    best_recall10 = -1.0
    best_state = None
    patience_counter = 0
    max_patience = 150

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for X, edge_index, edge_attr, mask, power_norm, power_raw, npw, nm in train_loader:
            X = X.to(device, non_blocking=True)
            edge_index = edge_index.to(device, non_blocking=True)
            if edge_attr is not None:
                edge_attr = edge_attr.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            power_norm = power_norm.to(device, non_blocking=True)
            power_raw = power_raw.to(device, non_blocking=True)
            npw = npw.to(device, non_blocking=True)
            nm = nm.to(device, non_blocking=True)

            optimizer.zero_grad()
            if w_node > 0:
                pred, node_pred = model(X, edge_index, mask, edge_attr=edge_attr, return_nodes=True)
            else:
                pred = model(X, edge_index, mask, edge_attr=edge_attr)
                node_pred = None

            loss_mse = huber_fn(pred, power_norm)
            loss_rank = compute_rank_loss(pred, power_raw, rank_fn, device)
            loss_list = listwise_loss(pred, power_norm)

            # Scale loss: 惩罚 pred 的 std 偏离 true 的 std (修正 Huber 压缩)
            # 用 unbiased=False 避免 batch_size 小时的 bias
            if pred.numel() >= 2:
                pred_std = pred.std(unbiased=False)
                true_std = power_norm.std(unbiased=False)
                loss_scale = (pred_std - true_std).abs()
            else:
                loss_scale = torch.tensor(0.0, device=device)

            # Node loss (Route D): warmup 后才加, 仅在有 mask 的节点上算
            if w_node > 0 and node_pred is not None and nm.any() and epoch >= node_warmup_epochs:
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
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / max(len(train_loader), 1)

        if (epoch + 1) % 5 == 0:
            model.eval()
            all_pred, all_true, all_n = [], [], []

            with torch.no_grad():
                for X, edge_index, edge_attr, mask, pn, pr, _npw, _nm in val_loader:
                    X = X.to(device, non_blocking=True)
                    edge_index = edge_index.to(device, non_blocking=True)
                    if edge_attr is not None:
                        edge_attr = edge_attr.to(device, non_blocking=True)
                    mask = mask.to(device, non_blocking=True)
                    pred = model(X, edge_index, mask, edge_attr=edge_attr)
                    pred_raw = pred * power_std + power_mean
                    all_pred.append(pred_raw)
                    all_true.append(pr.to(device, non_blocking=True))
                    all_n.append(mask.sum(dim=1).long())

            all_pred = torch.cat(all_pred)
            all_true = torch.cat(all_true)
            all_n = torch.cat(all_n)

            # Mixed-N 指标（被 N→power 短路污染，仅作参考）
            m_mix = _compute_metrics(all_pred, all_true)

            # Fixed-N 指标（去除 N 短路，真实路由学习能力）
            m_fix = None
            if eval_filter_n is not None:
                keep = (all_n == eval_filter_n)
                n_kept = int(keep.sum().item())
                if n_kept >= 16:
                    m_fix = _compute_metrics(all_pred[keep], all_true[keep])
                    m_fix["n_kept"] = n_kept

            line = (f"  [Fold {fold_id}] Epoch {epoch + 1:03d} | Loss: {avg_loss:.4f}"
                    f" | mix: τ={m_mix['tau']:+.3f} ρ={m_mix['rho']:+.3f} "
                    f"R@10={m_mix['recalls'][0.10]:.3f} MAPE={m_mix['mape']:.2f}%")
            if m_fix is not None:
                line += (f" || fix(N={eval_filter_n},n={m_fix['n_kept']}): "
                         f"τ={m_fix['tau']:+.3f} ρ={m_fix['rho']:+.3f} "
                         f"R@5={m_fix['recalls'][0.05]:.3f} "
                         f"R@10={m_fix['recalls'][0.10]:.3f} "
                         f"R@20={m_fix['recalls'][0.20]:.3f}")
            print(line)

            # best 选择标准：优先用 fixed-N R@10%（去除短路污染），fallback mixed-N
            track_metric = m_fix["recalls"][0.10] if m_fix is not None else m_mix["recalls"][0.10]
            tracked = m_fix if m_fix is not None else m_mix
            improved = track_metric > best_recall10
            if improved:
                best_recall10 = track_metric
                best_tau = tracked["tau"]
                best_spearman = tracked["rho"]
                patience_counter = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                tag = f"fix-N={eval_filter_n}" if m_fix is not None else "mix"
                print(f"       🌟 Fold {fold_id} best R@10% [{tag}]={best_recall10:.3f} "
                      f"(τ={best_tau:+.4f}, ρ={best_spearman:+.4f})")
            else:
                patience_counter += 5
                if patience_counter >= max_patience:
                    print(f"       ⏹  Fold {fold_id} 早停")
                    break

    return best_tau, best_state


# ========================== 主函数: K-Fold ==========================
def train_proxy(
    data_path: str = "dataset/glitch_power_data_16bit_enriched.pt",
    save_path: str = "dataset/glitch_power_proxy_gnn.pth",
    n_folds: int = 5,
    batch_size: int = 64,
    val_batch_size: int = 256,
    epochs: int = 500,
    lr: float = 3e-4,             # ← GNN 参数多，LR 调小一些
    weight_decay: float = 5e-3,   # ← 加大 weight decay 防过拟合
    node_feature_dim: int = 9,
    hidden_dim: int = 96,         # ← 64→96，增加表达力
    num_gnn_layers: int = 4,      # ← 3→4 层，扩大感受野
    dropout: float = 0.15,        # ← 略大于之前的 0.1
    w_mse: float = 0.5,       # ← 从 1.0 降到 0.5：Huber 压缩输出 scale，降权
    w_rank: float = 0.2,
    w_list: float = 0.5,
    w_scale: float = 0.5,     # ← 新增：惩罚 std(pred) 偏离 std(true)，对抗 Huber 压缩
    num_workers: int = 4,
    max_folds: int = 1,       # 只跑前 N 个 fold，便于快速对比实验
    start_fold: int = 0,      # 从第几个 fold 开始 (方便补跑剩余 fold)
    use_mean_agg: bool = True,   # GNN 用 mean 聚合（按 degree 归一化）
    use_edge_feat: bool = True,  # 用 arrival_time 差作为边特征
    eval_filter_n: int = 730,    # val 评估时只算 N==该值 的子集 (None=全集)
    train_filter_n: int = None,  # 训练时只用 N==该值 的子集 (C 方案: 消除 N 短路)
    use_stratified_batch: bool = False,  # batch 内同 N (B 方案: rank loss 不被 N 污染)
    external_edge_attr_dim: int = 0,     # v2 数据传入的 edge_attr 维度 (5 = sum/carry/port_a/b/c)
    w_node: float = 0.0,                 # Route D: per-FA node power 辅助损失权重
    node_warmup_epochs: int = 20,
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

    # 自动检测 v2 数据 (含 edge_attr) → 启用 external_edge_attr_dim
    sample_ea = dataset.data[0].get("edge_attr", None)
    if sample_ea is not None and external_edge_attr_dim == 0:
        external_edge_attr_dim = sample_ea.shape[-1]
        print(f"  ℹ  检测到 v2 edge_attr (dim={external_edge_attr_dim})，启用 external edge feature")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_taus = []
    best_overall_tau = -1.0
    best_overall_state = None

    # 参数量统计
    tmp_model = ArithProxyGNN(
        node_feature_dim, hidden_dim, num_gnn_layers, dropout=dropout,
        use_mean_agg=use_mean_agg, use_edge_feat=use_edge_feat,
        external_edge_attr_dim=external_edge_attr_dim,
    )
    n_params = sum(p.numel() for p in tmp_model.parameters())
    actual_use_edge_feat = tmp_model.use_edge_feat  # 可能被自动降级
    del tmp_model

    print(f"\n  🚀 开始 {n_folds}-Fold 交叉验证训练"
          + (f" (只跑前 {max_folds} 个 fold)" if max_folds else ""))
    print(f"     模型: ArithProxyGNN ({num_gnn_layers}-layer Bidirectional GNN")
    print(f"            + 异构投影 + 图级双池化)")
    print(f"     聚合方式: {'mean (degree-normalized)' if use_mean_agg else 'sum'}")
    print(f"     边特征  : {'arrival_skew + arrival_src' if actual_use_edge_feat else '关闭'}")
    print(f"     参数量: {n_params:,}")
    print(f"     hidden_dim={hidden_dim}, in_dim={node_feature_dim}, dropout={dropout}")
    print(f"     LR={lr}, weight_decay={weight_decay}")
    print(f"     Loss: {w_mse}×Huber + {w_rank}×RankLoss + {w_list}×ListMLE + {w_scale}×ScaleLoss")
    print(f"     评估 : best 选择基于 "
          + (f"fix-N={eval_filter_n} R@10%" if eval_filter_n else "mixed-N R@10%"))
    print(f"     训练数据: "
          + (f"只用 N={train_filter_n} 子集" if train_filter_n else "全集")
          + (" + stratified batch (同 N 同 batch)" if use_stratified_batch else ""))
    print(f"     DataLoader: train_bs={batch_size}, val_bs={val_batch_size}, num_workers={num_workers}\n")

    # POC: 预先算每个样本的 N 用于 filter / stratify
    sample_ns = np.array([dataset.data[i]["X"].shape[0] for i in range(len(dataset))])

    for fold_id, (train_idx, val_idx) in enumerate(kf.split(range(len(dataset)))):
        if fold_id < start_fold:
            continue
        if max_folds is not None and fold_id >= max_folds:
            print(f"\n  ⏭  达到 max_folds={max_folds}，提前结束")
            break

        # C 方案: 只保留 N==train_filter_n 的训练样本
        train_idx_full = train_idx
        if train_filter_n is not None:
            train_idx = train_idx[sample_ns[train_idx] == train_filter_n]
            print(f"  📌 train_filter_n={train_filter_n}: "
                  f"训练样本 {len(train_idx_full)} → {len(train_idx)}")

        print(f"  {'=' * 60}")
        print(f"  Fold {fold_id}: train={len(train_idx)}, val={len(val_idx)}")
        print(f"  {'=' * 60}")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        actual_bs = min(batch_size, len(train_idx))
        actual_val_bs = min(val_batch_size, len(val_idx))

        if use_stratified_batch:
            # B 方案: 自定义 BatchSampler 让 batch 内 N 一致
            train_batch_sampler = StratifiedNBatchSampler(
                train_subset, batch_size=actual_bs, drop_last=True,
                shuffle=True, seed=42 + fold_id,
            )
            train_loader = DataLoader(
                train_subset,
                batch_sampler=train_batch_sampler,
                collate_fn=custom_collate,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=(num_workers > 0),
            )
        else:
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
            use_mean_agg=use_mean_agg,
            use_edge_feat=use_edge_feat,
            external_edge_attr_dim=external_edge_attr_dim,
        ).to(device)

        tau, state = train_one_fold(
            model, train_loader, val_loader,
            device, epochs, lr, weight_decay,
            w_mse, w_rank, w_list, w_scale,
            power_mean, power_std, fold_id,
            eval_filter_n=eval_filter_n,
            w_node=w_node, node_warmup_epochs=node_warmup_epochs,
        )

        # 每个 fold 训练完立即保存，避免后续中断丢失
        fold_ckpt_path = save_path.replace(".pth", f"_fold{fold_id}.pth")
        torch.save({
            "model_state_dict": state,
            "node_feature_dim": node_feature_dim,
            "hidden_dim": hidden_dim,
            "num_gnn_layers": num_gnn_layers,
            "dropout": dropout,
            "power_mean": dataset.power_mean.item(),
            "power_std": dataset.power_std.item(),
            "model_class": "ArithProxyGNN",
            "use_mean_agg": use_mean_agg,
            "use_edge_feat": use_edge_feat,
            "fold_id": fold_id,
            "best_tau": tau,
        }, fold_ckpt_path)
        print(f"       💾 Fold {fold_id} ckpt 已保存: {fold_ckpt_path}")

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
        "use_mean_agg": use_mean_agg,
        "use_edge_feat": use_edge_feat,
    }, save_path)
    print(f"\n  ✅ 最佳模型已保存至: {save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["B", "C", "default"], default="default",
                        help="B=stratified batch (全集), C=train_filter_n=730, default=Run4 设定")
    parser.add_argument("--folds", type=int, default=1,
                        help="max_folds: 跑到第几个 fold 为止 (5 = 跑完全部)")
    parser.add_argument("--start_fold", type=int, default=0,
                        help="从第几个 fold 开始 (便于补跑剩余 fold)")
    parser.add_argument("--save_suffix", type=str, default=None,
                        help="ckpt 保存路径后缀 (避免覆盖现有 ckpt)")
    parser.add_argument("--data", type=str, default=None,
                        help="自定义训练数据路径 (默认 dataset/glitch_power_data_16bit_enriched.pt)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--w_node", type=float, default=0.0,
                        help="Route D: 节点级 power 辅助损失权重 (>0 启用)")
    args = parser.parse_args()

    kw = {}
    if args.data is not None:
        kw["data_path"] = args.data
    if args.epochs is not None:
        kw["epochs"] = args.epochs
    if args.w_node > 0:
        kw["w_node"] = args.w_node

    if args.mode == "C":
        save_path = f"dataset/glitch_power_proxy_gnn_C{args.save_suffix or ''}.pth"
        train_proxy(
            max_folds=args.folds, start_fold=args.start_fold,
            train_filter_n=730, use_stratified_batch=False,
            save_path=save_path, **kw,
        )
    elif args.mode == "B":
        save_path = f"dataset/glitch_power_proxy_gnn_B{args.save_suffix or ''}.pth"
        train_proxy(
            max_folds=args.folds, start_fold=args.start_fold,
            train_filter_n=None, use_stratified_batch=True,
            save_path=save_path, **kw,
        )
    else:
        train_proxy(max_folds=args.folds, start_fold=args.start_fold, **kw)