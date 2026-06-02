#!/usr/bin/env python3
"""Standalone Graphormer + one-hot proxy trainer.

This script intentionally does not import train_proxy.py. It reimplements dataset
loading, batching, losses, metrics, KFold training, and checkpointing locally.

Node input features are only X[:, onehot_start:onehot_start + onehot_dim].
Graph structure enters through Graphormer-style attention bias.
"""

import argparse
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import kendalltau, spearmanr
from sklearn.model_selection import KFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Sampler, Subset


# ----------------------------- data -----------------------------

class OneHotGraphDataset(Dataset):
    def __init__(self, data_path: str, target: str = "power"):
        raw = torch.load(data_path, map_location="cpu")
        if not raw:
            raise ValueError(f"empty dataset: {data_path}")

        if "edge_index" not in raw[0]:
            if "P" not in raw[0]:
                raise ValueError("dataset item has neither edge_index nor P")
            print("  converting dense P to sparse edge_index in memory")
            for item in raw:
                p = item.pop("P")
                item["edge_index"] = p.nonzero(as_tuple=False).t().contiguous().long()

        if target not in raw[0]:
            raise ValueError(f"dataset item lacks target '{target}', keys={list(raw[0].keys())}")

        self.data = raw
        self.target = target
        targets = torch.tensor([float(item[target]) for item in raw], dtype=torch.float32)
        self.target_mean = targets.mean()
        self.target_std = targets.std() + 1e-8

        feat_dim = int(raw[0]["X"].shape[1])
        n_edges_avg = sum(int(item["edge_index"].shape[1]) for item in raw) / len(raw)
        print(f"  dataset: {len(raw)} samples, X dim={feat_dim}, avg edges={n_edges_avg:.0f}")
        print(f"  target={target}: mean={self.target_mean:.6f}, std={self.target_std:.6f}, "
              f"min={targets.min():.6f}, max={targets.max():.6f}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        x = item["X"]
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        else:
            x = x.float()

        edge_index = item["edge_index"]
        if not torch.is_tensor(edge_index):
            edge_index = torch.tensor(edge_index, dtype=torch.long)
        else:
            edge_index = edge_index.long()

        y_raw = torch.tensor(float(item[self.target]), dtype=torch.float32)
        y_norm = (y_raw - self.target_mean) / self.target_std
        return {
            "x": x,
            "edge_index": edge_index,
            "y_norm": y_norm,
            "y_raw": y_raw,
            "n": x.shape[0],
        }


class StratifiedNBatchSampler(Sampler[List[int]]):
    def __init__(self, subset: Subset, batch_size: int, drop_last: bool = True,
                 shuffle: bool = True, seed: int = 42):
        self.subset = subset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        buckets: Dict[int, List[int]] = defaultdict(list)
        for local_idx, global_idx in enumerate(subset.indices):
            n = int(subset.dataset.data[int(global_idx)]["X"].shape[0])
            buckets[n].append(local_idx)
        self.buckets = dict(buckets)

        sizes = sorted(((n, len(v)) for n, v in self.buckets.items()), key=lambda x: x[1], reverse=True)
        top = ", ".join([f"N={n}:{c}" for n, c in sizes[:5]])
        print(f"  StratifiedNBatchSampler: {len(sizes)} N buckets, top5: {top}")

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        batches = []
        for _, idxs0 in self.buckets.items():
            idxs = list(idxs0)
            if self.shuffle:
                rng.shuffle(idxs)
            for i in range(0, len(idxs), self.batch_size):
                b = idxs[i:i + self.batch_size]
                if len(b) == self.batch_size or (len(b) > 0 and not self.drop_last):
                    batches.append(b)
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        n = 0
        for idxs in self.buckets.values():
            full = len(idxs) // self.batch_size
            rem = len(idxs) % self.batch_size
            n += full + (0 if self.drop_last or rem == 0 else 1)
        return n


def _adj_spatial_pos(n: int, edge_index: torch.Tensor) -> torch.Tensor:
    pos = torch.full((n, n), 2, dtype=torch.long)
    pos.fill_diagonal_(0)
    if edge_index.numel() > 0:
        src = edge_index[0].long().clamp(0, n - 1)
        dst = edge_index[1].long().clamp(0, n - 1)
        pos[src, dst] = 1
        pos[dst, src] = 1
    return pos


def _shortest_spatial_pos(n: int, edge_index: torch.Tensor, max_dist: int) -> torch.Tensor:
    # Buckets: 0 self, 1..max_dist shortest path, max_dist+1 unreachable/far.
    far = max_dist + 1
    adj = [[] for _ in range(n)]
    if edge_index.numel() > 0:
        srcs = edge_index[0].tolist()
        dsts = edge_index[1].tolist()
        for u, v in zip(srcs, dsts):
            if 0 <= u < n and 0 <= v < n and u != v:
                adj[u].append(v)
                adj[v].append(u)

    pos = torch.full((n, n), far, dtype=torch.long)
    for s in range(n):
        pos[s, s] = 0
        dist = [-1] * n
        dist[s] = 0
        queue = [s]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            du = dist[u]
            if du >= max_dist:
                continue
            nd = du + 1
            for v in adj[u]:
                if dist[v] < 0:
                    dist[v] = nd
                    pos[s, v] = nd
                    queue.append(v)
    return pos


@dataclass
class GraphormerCollator:
    spatial_mode: str = "adj"
    max_dist: int = 4

    def __call__(self, batch):
        bsz = len(batch)
        max_n = max(int(item["n"]) for item in batch)
        feat_dim = int(batch[0]["x"].shape[1])

        x_pad = torch.zeros(bsz, max_n, feat_dim, dtype=torch.float32)
        mask = torch.zeros(bsz, max_n, dtype=torch.bool)
        spatial = torch.zeros(bsz, max_n, max_n, dtype=torch.long)
        y_norm = torch.empty(bsz, dtype=torch.float32)
        y_raw = torch.empty(bsz, dtype=torch.float32)
        ns = torch.empty(bsz, dtype=torch.long)

        far_bucket = 2 if self.spatial_mode == "adj" else self.max_dist + 1
        spatial.fill_(far_bucket)

        for i, item in enumerate(batch):
            x = item["x"]
            n = int(item["n"])
            ei = item["edge_index"]
            x_pad[i, :n] = x
            mask[i, :n] = True
            ns[i] = n
            y_norm[i] = item["y_norm"]
            y_raw[i] = item["y_raw"]
            if self.spatial_mode == "shortest":
                sp = _shortest_spatial_pos(n, ei, self.max_dist)
            else:
                sp = _adj_spatial_pos(n, ei)
            spatial[i, :n, :n] = sp

        return x_pad, spatial, mask, y_norm, y_raw, ns


# ----------------------------- model -----------------------------

class GraphormerSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_bias, mask):
        bsz, n, hdim = x.shape
        qkv = self.qkv(x).view(bsz, n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # [B, H, N, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + attn_bias
        scores = scores.masked_fill(~mask[:, None, None, :], -1.0e4)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bsz, n, hdim)
        return self.out(out)


class GraphormerLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = GraphormerSelfAttention(hidden_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_bias, mask):
        x = x + self.dropout(self.attn(self.norm1(x), attn_bias, mask))
        x = x * mask.unsqueeze(-1).to(x.dtype)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        x = x * mask.unsqueeze(-1).to(x.dtype)
        return x


class GraphormerOneHot(nn.Module):
    def __init__(self, node_feature_dim: int, hidden_dim: int = 96, num_layers: int = 4,
                 num_heads: int = 4, ffn_dim: Optional[int] = None, dropout: float = 0.1,
                 onehot_start: int = 3, onehot_dim: int = 4, num_spatial_buckets: int = 3):
        super().__init__()
        if node_feature_dim < onehot_start + onehot_dim:
            raise ValueError("node_feature_dim is too small for requested onehot slice")
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_p = dropout
        self.onehot_start = onehot_start
        self.onehot_dim = onehot_dim
        self.num_spatial_buckets = num_spatial_buckets

        self.input_proj = nn.Linear(onehot_dim, hidden_dim)
        self.graph_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.spatial_bias = nn.Embedding(num_spatial_buckets, num_heads)
        self.layers = nn.ModuleList([
            GraphormerLayer(hidden_dim, num_heads, ffn_dim or hidden_dim * 4, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.graph_token, std=0.02)
        nn.init.normal_(self.spatial_bias.weight, std=0.02)

    def forward(self, x, spatial_pos, mask):
        bsz, n, _ = x.shape
        x_onehot = x[:, :, self.onehot_start:self.onehot_start + self.onehot_dim]
        h = self.input_proj(x_onehot)

        token = self.graph_token.expand(bsz, -1, -1)
        h = torch.cat([token, h], dim=1)
        token_mask = torch.ones(bsz, 1, dtype=torch.bool, device=mask.device)
        mask_ext = torch.cat([token_mask, mask], dim=1)

        spatial_pos = spatial_pos.clamp(0, self.num_spatial_buckets - 1).to(x.device)
        sp_ext = torch.zeros(bsz, n + 1, n + 1, dtype=torch.long, device=x.device)
        sp_ext[:, 1:, 1:] = spatial_pos
        bias = self.spatial_bias(sp_ext).permute(0, 3, 1, 2).contiguous()

        for layer in self.layers:
            h = layer(h, bias, mask_ext)
        graph_h = self.final_norm(h[:, 0])
        return self.head(graph_h).squeeze(-1)


# ----------------------------- losses / metrics -----------------------------

def compute_rank_loss(pred, true_raw, criterion, device):
    bsz = pred.size(0)
    if bsz < 2:
        return torch.tensor(0.0, device=device)
    idx_i, idx_j = torch.triu_indices(bsz, bsz, offset=1, device=device)
    max_pairs = 2048
    if idx_i.numel() > max_pairs:
        perm = torch.randperm(idx_i.numel(), device=device)[:max_pairs]
        idx_i, idx_j = idx_i[perm], idx_j[perm]
    diff = true_raw[idx_i] - true_raw[idx_j]
    valid = diff.abs() > 1e-8
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device)
    target = torch.sign(diff[valid])
    return criterion(pred[idx_i[valid]], pred[idx_j[valid]], target)


def listwise_loss(pred, true_norm):
    _, sorted_indices = true_norm.sort(descending=True)
    pred_sorted = pred[sorted_indices]
    pred_shifted = pred_sorted - pred_sorted.max()
    cumsums = torch.logcumsumexp(pred_shifted.flip(0), dim=0).flip(0)
    return (cumsums - pred_shifted).mean()


def compute_topk_recall(pred, true, ratios=(0.05, 0.10, 0.20)):
    out = {}
    n = pred.numel()
    for r in ratios:
        k = max(1, int(round(n * r)))
        true_top = set(torch.topk(true, k, largest=False).indices.cpu().tolist())
        pred_top = set(torch.topk(pred, k, largest=False).indices.cpu().tolist())
        out[r] = len(true_top & pred_top) / k
    return out


def compute_metrics(pred, true):
    p = pred.detach().cpu().numpy()
    t = true.detach().cpu().numpy()
    tau = kendalltau(p, t).correlation
    rho = spearmanr(p, t).correlation
    tau = 0.0 if np.isnan(tau) else float(tau)
    rho = 0.0 if np.isnan(rho) else float(rho)
    mse = F.mse_loss(pred, true).item()
    mape = (((pred - true).abs() / (true.abs() + 1e-8)).mean() * 100).item()
    return {"tau": tau, "rho": rho, "recalls": compute_topk_recall(pred, true), "mse": mse, "mape": mape}


def compute_stratified_metrics(pred, true, ns, min_bucket=16):
    vals = []
    for n in torch.unique(ns).tolist():
        keep = ns == n
        cnt = int(keep.sum().item())
        if cnt < min_bucket:
            continue
        m = compute_metrics(pred[keep], true[keep])
        vals.append((cnt, m))
    if not vals:
        return None
    weights = np.array([v[0] for v in vals], dtype=float)
    weights /= weights.sum()
    return {
        "tau": float(np.dot(weights, [v[1]["tau"] for v in vals])),
        "rho": float(np.dot(weights, [v[1]["rho"] for v in vals])),
        "r10": float(np.dot(weights, [v[1]["recalls"][0.10] for v in vals])),
        "n_buckets": len(vals),
        "n_total": int(sum(v[0] for v in vals)),
    }


@torch.no_grad()
def evaluate_model(model, loader, device, target_mean, target_std, eval_filter_n=None, amp=False):
    model.eval()
    preds, trues, ns_all = [], [], []
    for x, spatial, mask, _y_norm, y_raw, ns in loader:
        x = x.to(device, non_blocking=True)
        spatial = spatial.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            pred_norm = model(x, spatial, mask)
        pred_raw = pred_norm.float() * target_std + target_mean
        preds.append(pred_raw.cpu())
        trues.append(y_raw.float().cpu())
        ns_all.append(ns.cpu())
    pred = torch.cat(preds)
    true = torch.cat(trues)
    ns = torch.cat(ns_all)
    out = {
        "mix": compute_metrics(pred, true),
        "strat": compute_stratified_metrics(pred, true, ns),
        "fix": None,
    }
    if eval_filter_n is not None:
        keep = ns == eval_filter_n
        if int(keep.sum().item()) >= 16:
            out["fix"] = compute_metrics(pred[keep], true[keep])
            out["fix"]["n_kept"] = int(keep.sum().item())
    return out


# ----------------------------- train -----------------------------

def train_one_fold(model, train_loader, val_loader, test_loader, device, args,
                   target_mean, target_std, fold_id):
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    huber = nn.SmoothL1Loss()
    rank_fn = nn.MarginRankingLoss(margin=args.rank_margin)

    best_metric = -1.0
    best_tau = -1.0
    best_rho = -1.0
    best_state = None
    patience = 0

    # print(f"  [Fold {fold_id}] train batches/epoch={len(train_loader)}, "
    #       f"eval_every={args.eval_every}, log_every={args.log_every}", flush=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_steps = 0
        optimizer.zero_grad(set_to_none=True)

        for step, (x, spatial, mask, y_norm, y_raw, _ns) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            spatial = spatial.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            y_norm = y_norm.to(device, non_blocking=True)
            y_raw = y_raw.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                pred = model(x, spatial, mask)

            # Keep the memory-heavy Graphormer forward in AMP, but compute losses in
            # fp32. CUDA half backward for logcumsumexp is unavailable in this torch.
            pred_loss = pred.float()
            y_norm_loss = y_norm.float()
            y_raw_loss = y_raw.float()
            loss_huber = huber(pred_loss, y_norm_loss)
            loss_rank = compute_rank_loss(pred_loss, y_raw_loss, rank_fn, device)
            loss_list = listwise_loss(pred_loss, y_norm_loss)
            if pred_loss.numel() >= 2:
                loss_scale = (pred_loss.std(unbiased=False) - y_norm_loss.std(unbiased=False)).abs()
            else:
                loss_scale = torch.tensor(0.0, device=device)
            loss = (args.w_mse * loss_huber + args.w_rank * loss_rank
                    + args.w_list * loss_list + args.w_scale * loss_scale)
            loss_for_backward = loss / args.grad_accum_steps

            scaler.scale(loss_for_backward).backward()
            total_loss += float(loss.detach().cpu())
            n_steps += 1

            if args.log_every > 0 and ((step + 1) == 1 or (step + 1) % args.log_every == 0):
                running = total_loss / max(n_steps, 1)
                print(f"  [Fold {fold_id}] Epoch {epoch + 1:03d} "
                      f"step {step + 1:04d}/{len(train_loader)} "
                      f"loss={running:.4f}", flush=True)

            if (step + 1) % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        if n_steps % args.grad_accum_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()
        avg_loss = total_loss / max(n_steps, 1)

        if (epoch + 1) % args.eval_every == 0:
            ev = evaluate_model(model, val_loader, device, target_mean, target_std,
                                eval_filter_n=args.eval_filter_n, amp=args.amp)
            mix, strat, fix = ev["mix"], ev["strat"], ev["fix"]
            line = (f"  [Fold {fold_id}] Epoch {epoch + 1:03d} | Loss={avg_loss:.4f}"
                    f" | mix: tau={mix['tau']:+.3f} rho={mix['rho']:+.3f}"
                    f" R@10={mix['recalls'][0.10]:.3f} MAPE={mix['mape']:.2f}%")
            if strat is not None:
                line += (f" || perN(b={strat['n_buckets']}): tau={strat['tau']:+.3f}"
                         f" rho={strat['rho']:+.3f} R@10={strat['r10']:.3f}")
            if fix is not None:
                line += (f" || fix(N={args.eval_filter_n},n={fix['n_kept']}):"
                         f" tau={fix['tau']:+.3f} rho={fix['rho']:+.3f}"
                         f" R@10={fix['recalls'][0.10]:.3f}")
            print(line, flush=True)

            tracked = fix if fix is not None else mix
            metric = tracked["recalls"][0.10]
            if metric > best_metric:
                best_metric = metric
                best_tau = tracked["tau"]
                best_rho = tracked["rho"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
                tag = f"fix-N={args.eval_filter_n}" if fix is not None else "mix"
                print(f"       best R@10 [{tag}]={best_metric:.3f} "
                      f"(tau={best_tau:+.4f}, rho={best_rho:+.4f})", flush=True)
            else:
                patience += args.eval_every
                if patience >= args.patience:
                    print(f"       early stop at epoch {epoch + 1}", flush=True)
                    break

    test_metrics = None
    if best_state is not None and test_loader is not None:
        model.load_state_dict(best_state)
        test_metrics = evaluate_model(model, test_loader, device, target_mean, target_std,
                                      eval_filter_n=args.eval_filter_n, amp=args.amp)
        msg = f"  [Fold {fold_id}] TEST:"
        if test_metrics["strat"] is not None:
            st = test_metrics["strat"]
            msg += f" perN tau={st['tau']:+.4f} rho={st['rho']:+.4f} R@10={st['r10']:.3f}"
        if test_metrics["fix"] is not None:
            fx = test_metrics["fix"]
            msg += f" | fix-N={args.eval_filter_n} tau={fx['tau']:+.4f} R@10={fx['recalls'][0.10]:.3f}"
        print(msg, flush=True)
    return best_tau, best_state, test_metrics


def aggregate_test_metrics(metrics):
    if not metrics:
        return

    def agg(getter):
        vals = [getter(m) for m in metrics if getter(m) is not None]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), float("nan"))

    st_tau = agg(lambda m: m["strat"]["tau"] if m["strat"] else None)
    st_rho = agg(lambda m: m["strat"]["rho"] if m["strat"] else None)
    st_r10 = agg(lambda m: m["strat"]["r10"] if m["strat"] else None)
    print("  TEST summary:")
    print(f"     per-N weighted: tau={st_tau[0]:+.4f} +/- {st_tau[1]:.4f}"
          f" | rho={st_rho[0]:+.4f} +/- {st_rho[1]:.4f}"
          f" | R@10={st_r10[0]:.4f} +/- {st_r10[1]:.4f}")


def make_save_path(args):
    suffix = args.save_suffix or ""
    target_tag = "" if args.target == "power" else f"_{args.target}"
    mode_tag = args.mode if args.mode in {"B", "C"} else "default"
    return f"dataset/glitch_power_graphormer_onehot_{mode_tag}{target_tag}{suffix}.pth"


def train_graphormer(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    if args.target == "area" and args.eval_filter_n is not None:
        print("  target=area: fixed-N eval/model selection disabled; using mixed-N R@10")
        args.eval_filter_n = None

    if args.spatial_mode == "adj":
        args.num_spatial_buckets = 3
    else:
        args.num_spatial_buckets = args.max_dist + 2

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"  device: {device}")

    dataset = OneHotGraphDataset(args.data, target=args.target)
    feat_dim = int(dataset.data[0]["X"].shape[1])
    if feat_dim < args.onehot_start + args.onehot_dim:
        raise ValueError("X dim is too small for onehot slice")

    sample_ns = np.array([int(item["X"].shape[0]) for item in dataset.data])
    model_probe = GraphormerOneHot(
        node_feature_dim=feat_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        onehot_start=args.onehot_start,
        onehot_dim=args.onehot_dim,
        num_spatial_buckets=args.num_spatial_buckets,
    )
    n_params = sum(p.numel() for p in model_probe.parameters())
    del model_probe

    print("\n  GraphormerOneHot training")
    print(f"     input: only X[:, {args.onehot_start}:{args.onehot_start + args.onehot_dim}] one-hot")
    print(f"     spatial bias: {args.spatial_mode}" + (f" (max_dist={args.max_dist})" if args.spatial_mode == "shortest" else ""))
    print(f"     hidden={args.hidden_dim}, layers={args.num_layers}, heads={args.num_heads}, params={n_params:,}")
    print(f"     loss: {args.w_mse}*Huber + {args.w_rank}*Rank + {args.w_list}*ListMLE + {args.w_scale}*Scale")
    print(f"     batch_size={args.batch_size}, val_batch_size={args.val_batch_size}, amp={args.amp}")
    print("     model selection: " + (f"fix-N={args.eval_filter_n} R@10" if args.eval_filter_n else "mixed-N R@10"))

    save_path = args.save_path or make_save_path(args)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    fold_taus = []
    fold_tests = []
    best_overall_tau = -1.0
    best_overall_state = None

    collate = GraphormerCollator(spatial_mode=args.spatial_mode, max_dist=args.max_dist)

    for fold_id, (train_idx, val_idx) in enumerate(kf.split(range(len(dataset)))):
        if fold_id < args.start_fold:
            continue
        if args.only_fold is not None and fold_id != args.only_fold:
            continue
        if args.only_fold is None and fold_id >= args.folds:
            break

        if args.mode == "C":
            before = len(train_idx)
            train_idx = train_idx[sample_ns[train_idx] == args.train_filter_n]
            print(f"  train_filter_n={args.train_filter_n}: {before} -> {len(train_idx)}")

        print("\n  " + "=" * 60)
        print(f"  Fold {fold_id}: train={len(train_idx)}, val={len(val_idx)}")
        print("  " + "=" * 60)

        rng = np.random.RandomState(args.seed + 1000 + fold_id)
        perm = rng.permutation(len(val_idx))
        half = len(val_idx) // 2
        sel_idx = val_idx[perm[:half]]
        test_idx = val_idx[perm[half:]]
        print(f"     val split: selection={len(sel_idx)}, heldout-test={len(test_idx)}")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, sel_idx)
        test_subset = Subset(dataset, test_idx)

        if args.mode == "B":
            batch_sampler = StratifiedNBatchSampler(
                train_subset, batch_size=min(args.batch_size, len(train_subset)),
                drop_last=True, shuffle=True, seed=args.seed + fold_id,
            )
            train_loader = DataLoader(
                train_subset, batch_sampler=batch_sampler, collate_fn=collate,
                num_workers=args.num_workers, pin_memory=True,
                persistent_workers=args.num_workers > 0,
            )
        else:
            train_loader = DataLoader(
                train_subset, batch_size=min(args.batch_size, len(train_subset)), shuffle=True,
                drop_last=True, collate_fn=collate, num_workers=args.num_workers,
                pin_memory=True, persistent_workers=args.num_workers > 0,
            )

        val_loader = DataLoader(
            val_subset, batch_size=min(args.val_batch_size, max(len(val_subset), 1)), shuffle=False,
            collate_fn=collate, num_workers=args.num_workers, pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )
        test_loader = DataLoader(
            test_subset, batch_size=min(args.val_batch_size, max(len(test_subset), 1)), shuffle=False,
            collate_fn=collate, num_workers=args.num_workers, pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )

        model = GraphormerOneHot(
            node_feature_dim=feat_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            ffn_dim=args.ffn_dim,
            dropout=args.dropout,
            onehot_start=args.onehot_start,
            onehot_dim=args.onehot_dim,
            num_spatial_buckets=args.num_spatial_buckets,
        ).to(device)

        tau, state, test_metrics = train_one_fold(
            model, train_loader, val_loader, test_loader, device, args,
            dataset.target_mean.to(device), dataset.target_std.to(device), fold_id,
        )

        if test_metrics is not None:
            fold_tests.append(test_metrics)
        if state is None:
            print(f"     fold {fold_id}: no best state, skip save")
            continue

        fold_path = save_path.replace(".pth", f"_fold{fold_id}.pth")
        ckpt = {
            "model_state_dict": state,
            "model_class": "GraphormerOneHot",
            "target": args.target,
            "node_feature_dim": feat_dim,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "ffn_dim": args.ffn_dim,
            "dropout": args.dropout,
            "onehot_start": args.onehot_start,
            "onehot_dim": args.onehot_dim,
            "spatial_mode": args.spatial_mode,
            "max_dist": args.max_dist,
            "num_spatial_buckets": args.num_spatial_buckets,
            "target_mean": float(dataset.target_mean),
            "target_std": float(dataset.target_std),
            "fold_id": fold_id,
            "best_tau": tau,
            "use_onehot_only": True,
            "independent_trainer": "train_graphormer_onehot.py",
        }
        torch.save(ckpt, fold_path)
        print(f"     saved fold ckpt: {fold_path}")

        fold_taus.append(tau)
        if tau > best_overall_tau:
            best_overall_tau = tau
            best_overall_state = state

    if fold_taus:
        print("\n  K-Fold selection summary:")
        for i, tau in enumerate(fold_taus):
            print(f"     completed fold {i}: tau={tau:+.4f}")
        print(f"     mean tau={np.mean(fold_taus):+.4f} +/- {np.std(fold_taus):.4f}")
    aggregate_test_metrics(fold_tests)

    if args.only_fold is None and best_overall_state is not None:
        torch.save({
            "model_state_dict": best_overall_state,
            "model_class": "GraphormerOneHot",
            "target": args.target,
            "node_feature_dim": feat_dim,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "ffn_dim": args.ffn_dim,
            "dropout": args.dropout,
            "onehot_start": args.onehot_start,
            "onehot_dim": args.onehot_dim,
            "spatial_mode": args.spatial_mode,
            "max_dist": args.max_dist,
            "num_spatial_buckets": args.num_spatial_buckets,
            "target_mean": float(dataset.target_mean),
            "target_std": float(dataset.target_std),
            "best_tau": best_overall_tau,
            "use_onehot_only": True,
            "independent_trainer": "train_graphormer_onehot.py",
        }, save_path)
        print(f"\n  saved aggregate best ckpt: {save_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Standalone Graphormer + one-hot proxy trainer")
    p.add_argument("--data", default="dataset/glitch_power_data_16bit_v2_13k_enriched.pt")
    p.add_argument("--target", choices=["power", "area", "delay"], default="power")
    p.add_argument("--mode", choices=["B", "C", "default"], default="B",
                   help="B=stratified same-N batches; C=train only N=train_filter_n; default=random batches")
    p.add_argument("--folds", type=int, default=1, help="number of folds to run from start_fold")
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--start_fold", type=int, default=0)
    p.add_argument("--only_fold", type=int, default=None)
    p.add_argument("--train_filter_n", type=int, default=730)
    p.add_argument("--eval_filter_n", type=int, default=730, help="<=0 disables; always disabled for target=area")
    p.add_argument("--save_path", default=None)
    p.add_argument("--save_suffix", default="")

    p.add_argument("--hidden_dim", type=int, default=96)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--ffn_dim", type=int, default=None)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--onehot_start", type=int, default=3)
    p.add_argument("--onehot_dim", type=int, default=4)
    p.add_argument("--spatial_mode", choices=["adj", "shortest"], default="adj")
    p.add_argument("--max_dist", type=int, default=4)

    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--val_batch_size", type=int, default=4)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eta_min", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=5e-3)
    p.add_argument("--w_mse", type=float, default=0.5)
    p.add_argument("--w_rank", type=float, default=0.2)
    p.add_argument("--w_list", type=float, default=0.5)
    p.add_argument("--w_scale", type=float, default=0.5)
    p.add_argument("--rank_margin", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=300)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--log_every", type=int, default=0,
                   help="print train progress every N batches; 0 disables (默认: 关闭, 只看 eval 性能)")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true", help="mixed precision for attention memory/speed")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    if args.eval_filter_n <= 0:
        args.eval_filter_n = None
    if args.ffn_dim is None:
        args.ffn_dim = args.hidden_dim * 4
    if args.batch_size < 1 or args.val_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    if args.hidden_dim % args.num_heads != 0:
        raise ValueError("hidden_dim must be divisible by num_heads")
    if args.grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be >= 1")
    return args


if __name__ == "__main__":
    train_graphormer(parse_args())
