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

from proxy_mlp import ArithProxyGNN, PureGIN, PureGCN     # ← 新 GNN + 纯 GIN/GCN 对照


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

    def __init__(self, data_path, target="area"):
        """
        Args:
            target: 主预测目标, "area" / "power" / "delay" (默认 area)
                    内部变量名保留 power_mean/std/norm/raw (历史代码兼容),
                    但实际承载的是 target 数据
        """
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

        if target not in raw[0]:
            raise ValueError(f"数据集 item 缺少 '{target}' 字段, 可用键: {list(raw[0].keys())}")

        self.data = raw
        self.target = target

        powers = torch.tensor([item[target] for item in raw], dtype=torch.float32)
        self.power_mean = powers.mean()
        self.power_std = powers.std() + 1e-8

        # 方案 2: area / delay 辅助目标的归一化常数 (整集统计, fold 间不漂移)
        if "area" in raw[0] and "delay" in raw[0]:
            areas = torch.tensor([item["area"] for item in raw], dtype=torch.float32)
            delays = torch.tensor([item["delay"] for item in raw], dtype=torch.float32)
            self.area_mean = areas.mean()
            self.area_std = areas.std() + 1e-8
            self.delay_mean = delays.mean()
            self.delay_std = delays.std() + 1e-8
            self.has_aux = True
        else:
            self.area_mean = torch.tensor(0.0)
            self.area_std = torch.tensor(1.0)
            self.delay_mean = torch.tensor(0.0)
            self.delay_std = torch.tensor(1.0)
            self.has_aux = False

        feat_dim = raw[0]["X"].shape[1]
        n_edges_avg = sum(item["edge_index"].shape[1] for item in raw) / len(raw)
        print(f"  📊 数据集: {len(raw)} 样本，X 维度={feat_dim}")
        print(f"     平均边数: {n_edges_avg:.0f}")
        print(f"  🎯 主目标 target={target}")
        print(f"     {target}: mean={self.power_mean:.6f}, std={self.power_std:.6f}, "
              f"min={powers.min():.6f}, max={powers.max():.6f}")
        if self.has_aux:
            print(f"     [aux] area : mean={self.area_mean:.3f}, std={self.area_std:.3f}")
            print(f"     [aux] delay: mean={self.delay_mean:.4f}, std={self.delay_std:.4f}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        X = item["X"]
        edge_index = item["edge_index"]
        # v2 数据带 edge_attr; 旧数据没有 → 返回 None 占位
        edge_attr = item.get("edge_attr", None)
        # 主目标 (变量名仍为 power_* 以保持下游兼容, 实际承载 self.target)
        power_val = item[self.target]
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
        # 方案 2: area / delay 辅助目标 (归一化后)
        if self.has_aux:
            area_norm = torch.tensor(
                (item["area"] - self.area_mean.item()) / self.area_std.item(),
                dtype=torch.float32,
            )
            delay_norm = torch.tensor(
                (item["delay"] - self.delay_mean.item()) / self.delay_std.item(),
                dtype=torch.float32,
            )
        else:
            area_norm = torch.tensor(0.0)
            delay_norm = torch.tensor(0.0)
        return (X, edge_index, edge_attr, power_norm, power_raw,
                node_powers, node_mask, area_norm, delay_norm)


# ========================== Collate ==========================
def custom_collate(batch):
    (X_list, ei_list, ea_list, pn_list, pr_list,
     np_list, nm_list, ar_list, dl_list) = zip(*batch)
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
    area_norm = torch.stack(ar_list)
    delay_norm = torch.stack(dl_list)
    return (X_pad, edge_index, edge_attr, mask,
            power_norm, power_raw, npw_pad, nmask_pad,
            area_norm, delay_norm)


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


def _compute_stratified_metrics(pred, true, ns, min_bucket=16):
    """Per-N 分层评估: 对每个 N 桶单独算 τ/ρ/R@10, 再按样本量加权平均。

    替代被 N→power 短路污染的 mixed-N 指标, 又比只看 N=730 用到了全部 val 样本,
    是衡量"跨结构家族真实排序能力"的诚实指标。
    """
    if ns is None:
        return None
    taus, rhos, r10s, weights = [], [], [], []
    for n in torch.unique(ns).tolist():
        m = (ns == n)
        cnt = int(m.sum().item())
        if cnt < min_bucket:
            continue
        mt = _compute_metrics(pred[m], true[m])
        taus.append(mt["tau"])
        rhos.append(mt["rho"])
        r10s.append(mt["recalls"][0.10])
        weights.append(cnt)
    if not taus:
        return None
    w = np.array(weights, dtype=float)
    w /= w.sum()
    return {
        "tau": float(np.dot(w, taus)),         # 样本量加权
        "rho": float(np.dot(w, rhos)),
        "r10": float(np.dot(w, r10s)),
        "tau_simple": float(np.mean(taus)),    # 等权 (每个 N 桶一票)
        "n_buckets": len(taus),
        "n_total": int(sum(weights)),
    }


def _eval_model(model, loader, device, power_mean, power_std, eval_filter_n=None):
    """在一个 loader 上跑前向, 返回 mixed-N / per-N 分层 / fix-N 三套指标。"""
    model.eval()
    all_pred, all_true, all_n = [], [], []
    with torch.no_grad():
        for (X, edge_index, edge_attr, mask, _pn, pr,
             _npw, _nm, _ar, _dl) in loader:
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

    out = {
        "mix": _compute_metrics(all_pred, all_true),
        "strat": _compute_stratified_metrics(all_pred, all_true, all_n),
        "fix": None,
    }
    if eval_filter_n is not None:
        keep = (all_n == eval_filter_n)
        if int(keep.sum().item()) >= 16:
            out["fix"] = _compute_metrics(all_pred[keep], all_true[keep])
            out["fix"]["n_kept"] = int(keep.sum().item())
    return out


def train_one_fold(
    model, train_loader, val_loader,
    device, epochs, lr, weight_decay,
    w_mse, w_rank, w_list, w_scale,
    power_mean, power_std, fold_id=0,
    eval_filter_n=None,
    w_node=0.0, node_warmup_epochs=20,
    use_multitask=False, w_area=0.0, w_delay=0.0,
    test_loader=None,
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
    max_patience = 300

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for (X, edge_index, edge_attr, mask, power_norm, power_raw,
             npw, nm, area_norm, delay_norm) in train_loader:
            X = X.to(device, non_blocking=True)
            edge_index = edge_index.to(device, non_blocking=True)
            if edge_attr is not None:
                edge_attr = edge_attr.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            power_norm = power_norm.to(device, non_blocking=True)
            power_raw = power_raw.to(device, non_blocking=True)
            npw = npw.to(device, non_blocking=True)
            nm = nm.to(device, non_blocking=True)
            area_norm = area_norm.to(device, non_blocking=True)
            delay_norm = delay_norm.to(device, non_blocking=True)

            optimizer.zero_grad()
            # 4 种组合: (w_node>0, use_multitask) ∈ {(F,F), (T,F), (F,T), (T,T)}
            area_pred = delay_pred = None
            if w_node > 0 and use_multitask:
                pred, node_pred, area_pred, delay_pred = model(
                    X, edge_index, mask, edge_attr=edge_attr,
                    return_nodes=True, return_aux=True,
                )
            elif w_node > 0:
                pred, node_pred = model(X, edge_index, mask, edge_attr=edge_attr, return_nodes=True)
            elif use_multitask:
                pred, area_pred, delay_pred = model(
                    X, edge_index, mask, edge_attr=edge_attr, return_aux=True,
                )
                node_pred = None
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

            # 方案 2: area / delay 辅助 huber loss (共享 backbone)
            if use_multitask and area_pred is not None:
                loss_area = huber_fn(area_pred, area_norm)
                loss_delay = huber_fn(delay_pred, delay_norm)
            else:
                loss_area = torch.tensor(0.0, device=device)
                loss_delay = torch.tensor(0.0, device=device)

            loss = (w_mse * loss_mse + w_rank * loss_rank
                    + w_list * loss_list + w_scale * loss_scale
                    + w_node_eff * loss_node
                    + w_area * loss_area + w_delay * loss_delay)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / max(len(train_loader), 1)

        if (epoch + 1) % 5 == 0:
            ev = _eval_model(model, val_loader, device,
                             power_mean, power_std, eval_filter_n)
            m_mix, m_fix, m_str = ev["mix"], ev["fix"], ev["strat"]

            line = (f"  [Fold {fold_id}] Epoch {epoch + 1:03d} | Loss: {avg_loss:.4f}"
                    f" | mix: τ={m_mix['tau']:+.3f} ρ={m_mix['rho']:+.3f} "
                    f"R@10={m_mix['recalls'][0.10]:.3f} MAPE={m_mix['mape']:.2f}%")
            if m_str is not None:
                line += (f" || perN(b={m_str['n_buckets']}): "
                         f"τ={m_str['tau']:+.3f} ρ={m_str['rho']:+.3f} "
                         f"R@10={m_str['r10']:.3f}")
            if m_fix is not None:
                line += (f" || fix(N={eval_filter_n},n={m_fix['n_kept']}): "
                         f"τ={m_fix['tau']:+.3f} ρ={m_fix['rho']:+.3f} "
                         f"R@5={m_fix['recalls'][0.05]:.3f} "
                         f"R@10={m_fix['recalls'][0.10]:.3f} "
                         f"R@20={m_fix['recalls'][0.20]:.3f}")
            print(line)

            # best 选择标准：优先用 fixed-N R@10%（去除短路污染），fallback mixed-N
            # 注意：该选择只用 val_loader，test_loader 永不参与选择 → 无偏估计
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

    # ===== 用从不参与模型选择的 test 集做无偏泛化评估 =====
    test_metrics = None
    if test_loader is not None and best_state is not None:
        model.load_state_dict(best_state)
        test_metrics = _eval_model(model, test_loader, device,
                                   power_mean, power_std, eval_filter_n)
        ts, tf = test_metrics["strat"], test_metrics["fix"]
        msg = f"  [Fold {fold_id}] 🧪 TEST (held-out, 未参与选择):"
        if ts is not None:
            msg += (f" perN τ={ts['tau']:+.4f} ρ={ts['rho']:+.4f} "
                    f"R@10={ts['r10']:.3f} (b={ts['n_buckets']}, n={ts['n_total']})")
        if tf is not None:
            msg += (f" | fix-N={eval_filter_n}(n={tf['n_kept']}) "
                    f"τ={tf['tau']:+.4f} R@10={tf['recalls'][0.10]:.3f}")
        print(msg)

    return best_tau, best_state, test_metrics


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
    # 方案 2/3/4 新增 flags
    use_typed_edges: bool = False,       # 方案 3: RGCN-style 边类型 embedding
    use_multitask: bool = False,         # 方案 2: 多任务 area+delay
    use_jk_pool: bool = False,           # 方案 4: JK-Net 多尺度池化
    use_gin: bool = False,               # 替换 backbone 为 BidirectionalGIN
    use_pure_gin: bool = False,          # 完全用纯 GIN (Xu et al. 2019), 关掉所有工程加强项
    use_pure_gcn: bool = False,          # 完全用纯 GCN (Kipf & Welling 2017)
    w_area: float = 0.2,                 # 方案 2: area 辅助 loss 权重
    w_delay: float = 0.2,                # 方案 2: delay 辅助 loss 权重
    target: str = "area",                # 主预测目标 (默认 area, 可选 power/delay)
    skip_aggregate_save: bool = False,   # --only_fold 模式: 跳过最终聚合 save 避免多进程竞争
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  🖥  Device: {device}")

    if target == "area" and eval_filter_n is not None:
        print("  ℹ  target=area: 关闭 fixed-N 评估/选模，使用 mixed-N R@10%")
        eval_filter_n = None

    dataset = ArithDataset(data_path, target=target)
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
    fold_test_metrics = []
    best_overall_tau = -1.0
    best_overall_state = None

    # 参数量统计
    if use_pure_gcn:
        tmp_model = PureGCN(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_gnn_layers=num_gnn_layers,
            dropout=dropout,
        )
    elif use_pure_gin:
        tmp_model = PureGIN(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_gnn_layers=num_gnn_layers,
            dropout=dropout,
        )
    else:
        tmp_model = ArithProxyGNN(
            node_feature_dim, hidden_dim, num_gnn_layers, dropout=dropout,
            use_mean_agg=use_mean_agg, use_edge_feat=use_edge_feat,
            external_edge_attr_dim=external_edge_attr_dim,
            use_typed_edges=use_typed_edges,
            use_multitask=use_multitask,
            use_jk_pool=use_jk_pool,
            use_gin=use_gin,
        )
    n_params = sum(p.numel() for p in tmp_model.parameters())
    actual_use_edge_feat = getattr(tmp_model, "use_edge_feat", False)  # PureGIN 没有该属性
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
    print(f"     Loss: {w_mse}×Huber + {w_rank}×RankLoss + {w_list}×ListMLE + {w_scale}×ScaleLoss"
          + (f" + {w_area}×AreaHuber + {w_delay}×DelayHuber" if use_multitask else ""))
    print(f"     方案 1 (edge_attr={external_edge_attr_dim}d): "
          + ("启用" if external_edge_attr_dim > 5 else "默认 5d / 关闭"))
    print(f"     方案 2 multitask area+delay     : {'启用' if use_multitask else '关闭'}")
    print(f"     方案 3 typed-edge embedding     : {'启用' if use_typed_edges else '关闭'}")
    print(f"     方案 4 JK-Net multi-scale pool  : {'启用' if use_jk_pool else '关闭'}")
    print(f"     GIN backbone                    : {'启用 (BidirectionalGIN)' if use_gin else '关闭 (BidirectionalGNN)'}")
    if use_pure_gin:
        print(f"     ⚡ PureGIN (Xu et al. 2019)     : 启用 — 覆盖以上所有架构选项")
    if use_pure_gcn:
        print(f"     ⚡ PureGCN (Kipf & Welling 2017): 启用 — 覆盖以上所有架构选项")
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

        # val_idx 再切 50/50 → 选择集 (early-stop/best) + test 集 (仅最终汇报, 永不选择)
        # 这样 test 既未参与训练, 也未参与模型选择 → 给出无偏泛化估计
        rng = np.random.RandomState(1234 + fold_id)
        perm = rng.permutation(len(val_idx))
        half = len(val_idx) // 2
        sel_idx = val_idx[perm[:half]]
        test_idx = val_idx[perm[half:]]
        print(f"     val 拆分: 选择集={len(sel_idx)}, test集(无偏)={len(test_idx)}")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, sel_idx)
        test_subset = Subset(dataset, test_idx)

        actual_bs = min(batch_size, len(train_idx))
        actual_val_bs = min(val_batch_size, len(sel_idx))

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
        test_loader = DataLoader(
            test_subset,
            batch_size=min(val_batch_size, max(len(test_idx), 1)),
            shuffle=False,
            collate_fn=custom_collate,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )

        if use_pure_gcn:
            model = PureGCN(
                node_feature_dim=node_feature_dim,
                hidden_dim=hidden_dim,
                num_gnn_layers=num_gnn_layers,
                dropout=dropout,
            ).to(device)
        elif use_pure_gin:
            model = PureGIN(
                node_feature_dim=node_feature_dim,
                hidden_dim=hidden_dim,
                num_gnn_layers=num_gnn_layers,
                dropout=dropout,
            ).to(device)
        else:
            model = ArithProxyGNN(
                node_feature_dim=node_feature_dim,
                hidden_dim=hidden_dim,
                num_gnn_layers=num_gnn_layers,
                dropout=dropout,
                use_mean_agg=use_mean_agg,
                use_edge_feat=use_edge_feat,
                external_edge_attr_dim=external_edge_attr_dim,
                use_typed_edges=use_typed_edges,
                use_multitask=use_multitask,
                use_jk_pool=use_jk_pool,
                use_gin=use_gin,
            ).to(device)

        tau, state, test_metrics = train_one_fold(
            model, train_loader, val_loader,
            device, epochs, lr, weight_decay,
            w_mse, w_rank, w_list, w_scale,
            power_mean, power_std, fold_id,
            eval_filter_n=eval_filter_n,
            w_node=w_node, node_warmup_epochs=node_warmup_epochs,
            use_multitask=use_multitask, w_area=w_area, w_delay=w_delay,
            test_loader=test_loader,
        )
        if test_metrics is not None:
            fold_test_metrics.append(test_metrics)

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
            "target": target,
            "model_class": "ArithProxyGNN",
            "use_mean_agg": use_mean_agg,
            "use_edge_feat": use_edge_feat,
            "external_edge_attr_dim": external_edge_attr_dim,
            "use_typed_edges": use_typed_edges,
            "use_multitask": use_multitask,
            "use_jk_pool": use_jk_pool,
            "use_gin": use_gin,
            "use_pure_gin": use_pure_gin,
            "use_pure_gcn": use_pure_gcn,
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
    print(f"  📊 K-Fold 结果 (val 选择集峰值, 乐观上界):")
    for i, t in enumerate(fold_taus):
        print(f"     Fold {i}: τ = {t:+.4f}")
    print(f"     平均 τ = {np.mean(fold_taus):+.4f} ± {np.std(fold_taus):.4f}")
    print(f"     最佳 τ = {best_overall_tau:+.4f}")
    print(f"  {'=' * 60}")

    # 无偏泛化汇总：test 集 (未参与训练, 也未参与模型选择)
    if fold_test_metrics:
        def _agg(getter):
            vals = [getter(m) for m in fold_test_metrics if getter(m) is not None]
            return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), float("nan"))

        st_tau = _agg(lambda m: m["strat"]["tau"] if m["strat"] else None)
        st_rho = _agg(lambda m: m["strat"]["rho"] if m["strat"] else None)
        st_r10 = _agg(lambda m: m["strat"]["r10"] if m["strat"] else None)
        print(f"  🧪 TEST 无偏泛化 (主指标, 对外汇报用):")
        print(f"     per-N 加权 : τ = {st_tau[0]:+.4f} ± {st_tau[1]:.4f}"
              f" | ρ = {st_rho[0]:+.4f} ± {st_rho[1]:.4f}"
              f" | R@10 = {st_r10[0]:.4f} ± {st_r10[1]:.4f}")
        if eval_filter_n is not None:
            fx_tau = _agg(lambda m: m["fix"]["tau"] if m["fix"] else None)
            fx_r10 = _agg(lambda m: m["fix"]["recalls"][0.10] if m["fix"] else None)
            print(f"     fix-N={eval_filter_n}  : τ = {fx_tau[0]:+.4f} ± {fx_tau[1]:.4f}"
                  f" | R@10 = {fx_r10[0]:.4f} ± {fx_r10[1]:.4f}")
        print(f"  {'=' * 60}")

    # --only_fold 模式: 跳过聚合 save 防多进程竞争 (per-fold ckpt 已在 fold 结束时单独保存)
    if skip_aggregate_save:
        print(f"\n  ℹ  --only_fold 模式: 跳过聚合 save, per-fold ckpt 已保存")
    else:
        torch.save({
            "model_state_dict": best_overall_state,
            "node_feature_dim": node_feature_dim,
            "hidden_dim": hidden_dim,
            "num_gnn_layers": num_gnn_layers,
            "dropout": dropout,
            "power_mean": dataset.power_mean.item(),
            "power_std": dataset.power_std.item(),
            "target": target,
            "model_class": "ArithProxyGNN",
            "use_mean_agg": use_mean_agg,
            "use_edge_feat": use_edge_feat,
            "external_edge_attr_dim": external_edge_attr_dim,
            "use_typed_edges": use_typed_edges,
            "use_multitask": use_multitask,
            "use_jk_pool": use_jk_pool,
            "use_gin": use_gin,
            "use_pure_gin": use_pure_gin,
            "use_pure_gcn": use_pure_gcn,
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
    parser.add_argument("--only_fold", type=int, default=None,
                        help="只跑指定的单个 fold (用于外部并行多进程跑 5 fold)")
    parser.add_argument("--save_suffix", type=str, default=None,
                        help="ckpt 保存路径后缀 (避免覆盖现有 ckpt)")
    parser.add_argument("--data", type=str, default=None,
                        help="自定义训练数据路径 (默认 dataset/glitch_power_data_16bit_enriched.pt)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None,
                        help="GNN hidden dimension (默认使用 train_proxy() 内置值)")
    parser.add_argument("--num_gnn_layers", type=int, default=None,
                        help="GNN 层数 (默认使用 train_proxy() 内置值)")
    parser.add_argument("--w_node", type=float, default=0.0,
                        help="Route D: 节点级 power 辅助损失权重 (>0 启用)")
    # 方案 2/3/4
    parser.add_argument("--use_typed_edges", action="store_true",
                        help="方案 3: RGCN-style 边类型 embedding (需要 edge_attr 含 sum/carry/port_a/b/c)")
    parser.add_argument("--use_multitask", action="store_true",
                        help="方案 2: 共享 backbone, 额外 area/delay head (huber loss 加权 w_area, w_delay)")
    parser.add_argument("--use_jk_pool", action="store_true",
                        help="方案 4: JK-Net 多尺度池化 — attention 加权融合每层 GNN h")
    parser.add_argument("--use_gin", action="store_true",
                        help="把 backbone 换成 GIN (BidirectionalGINLayer: sum agg + 学 ε + MLP)")
    parser.add_argument("--use_pure_gin", action="store_true",
                        help="完全用纯 GIN (Xu et al. 2019) 模型, 覆盖所有架构 flag")
    parser.add_argument("--use_pure_gcn", action="store_true",
                        help="完全用纯 GCN (Kipf & Welling 2017) 模型, 覆盖所有架构 flag")
    parser.add_argument("--w_area", type=float, default=0.2,
                        help="方案 2: area 辅助 huber loss 权重 (默认 0.2)")
    parser.add_argument("--w_delay", type=float, default=0.2,
                        help="方案 2: delay 辅助 huber loss 权重 (默认 0.2)")
    parser.add_argument("--target", choices=["area", "power", "delay"], default="area",
                        help="主预测目标 (默认 area)")
    parser.add_argument("--eval_filter_n", type=int, default=730,
                        help="fix-N 子集 (默认 730 是 16-bit; 8-bit 用 369)")
    args = parser.parse_args()

    kw = {"target": args.target}
    if args.data is not None:
        kw["data_path"] = args.data
    if args.epochs is not None:
        kw["epochs"] = args.epochs
    if args.hidden_dim is not None:
        kw["hidden_dim"] = args.hidden_dim
    if args.num_gnn_layers is not None:
        kw["num_gnn_layers"] = args.num_gnn_layers
    if args.w_node > 0:
        kw["w_node"] = args.w_node
    if args.use_typed_edges:
        kw["use_typed_edges"] = True
    if args.use_multitask:
        kw["use_multitask"] = True
        kw["w_area"] = args.w_area
        kw["w_delay"] = args.w_delay
    if args.use_jk_pool:
        kw["use_jk_pool"] = True
    if args.use_gin:
        kw["use_gin"] = True
    if args.use_pure_gin:
        kw["use_pure_gin"] = True
    if args.use_pure_gcn:
        kw["use_pure_gcn"] = True
    kw["eval_filter_n"] = args.eval_filter_n if args.eval_filter_n > 0 else None

    # --only_fold N: 只跑单个 fold (多进程外部并行用)
    # 转成 start_fold=N, max_folds=N+1, 并禁用聚合 save 防多进程竞争
    if args.only_fold is not None:
        kw["start_fold"] = args.only_fold
        kw["max_folds"] = args.only_fold + 1
        kw["skip_aggregate_save"] = True
    else:
        kw["start_fold"] = args.start_fold
        kw["max_folds"] = args.folds

    # 路径里带 target 后缀, 避免覆盖不同目标的 ckpt
    tgt_tag = "" if args.target == "power" else f"_{args.target}"
    if args.mode == "C":
        save_path = f"dataset/glitch_power_proxy_gnn_C{tgt_tag}{args.save_suffix or ''}.pth"
        train_proxy(
            train_filter_n=730, use_stratified_batch=False,
            save_path=save_path, **kw,
        )
    elif args.mode == "B":
        save_path = f"dataset/glitch_power_proxy_gnn_B{tgt_tag}{args.save_suffix or ''}.pth"
        train_proxy(
            train_filter_n=None, use_stratified_batch=True,
            save_path=save_path, **kw,
        )
    else:
        train_proxy(**kw)