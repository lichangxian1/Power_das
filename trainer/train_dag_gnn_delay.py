#!/usr/bin/env python3
"""Train a DAG-aware GNN proxy for post-synthesis delay prediction."""

import argparse
import os
import random
import sys

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from out_dir_util import resolve_out_dir, place
from proxy_mlp import DAGTimingGNN
from train_proxy import (
    ArithDataset,
    StratifiedNBatchSampler,
    custom_collate,
    train_one_fold,
)


def _aggregate_test_metrics(fold_test_metrics, eval_filter_n):
    if not fold_test_metrics:
        return

    def agg(getter):
        vals = [getter(m) for m in fold_test_metrics if getter(m) is not None]
        if not vals:
            return float("nan"), float("nan")
        return float(np.mean(vals)), float(np.std(vals))

    st_tau = agg(lambda m: m["strat"]["tau"] if m["strat"] else None)
    st_rho = agg(lambda m: m["strat"]["rho"] if m["strat"] else None)
    st_r10 = agg(lambda m: m["strat"]["r10"] if m["strat"] else None)
    print("  TEST unbiased generalization:")
    print(f"     per-N weighted: tau={st_tau[0]:+.4f} +/- {st_tau[1]:.4f}"
          f" | rho={st_rho[0]:+.4f} +/- {st_rho[1]:.4f}"
          f" | R@10={st_r10[0]:.4f} +/- {st_r10[1]:.4f}")
    if eval_filter_n is not None:
        fx_tau = agg(lambda m: m["fix"]["tau"] if m["fix"] else None)
        fx_r10 = agg(lambda m: m["fix"]["recalls"][0.10] if m["fix"] else None)
        print(f"     fix-N={eval_filter_n}: tau={fx_tau[0]:+.4f} +/- {fx_tau[1]:.4f}"
              f" | R@10={fx_r10[0]:.4f} +/- {fx_r10[1]:.4f}")


def _save_ckpt(path, state, dataset, model_kwargs, target, fold_id=None, best_tau=None):
    payload = {
        "model_state_dict": state,
        "model_class": "DAGTimingGNN",
        "target": target,
        "power_mean": dataset.power_mean.item(),
        "power_std": dataset.power_std.item(),
        "best_tau": best_tau,
        **model_kwargs,
    }
    if fold_id is not None:
        payload["fold_id"] = fold_id
    torch.save(payload, path)


def train_dag_gnn_delay(
    data_path="dataset/glitch_power_data_16bit_v2_13k_edge10.pt",
    save_path="dataset/glitch_power_dag_gnn_delay.pth",
    n_splits=5,
    max_folds=1,
    start_fold=0,
    batch_size=64,
    val_batch_size=256,
    epochs=300,
    lr=3e-4,
    weight_decay=5e-3,
    hidden_dim=96,
    num_gnn_layers=4,
    dropout=0.10,
    w_mse=0.7,
    w_rank=0.2,
    w_list=0.3,
    w_scale=0.3,
    num_workers=4,
    eval_filter_n=730,
    train_filter_n=None,
    use_stratified_batch=True,
    target="delay",
    topo_idx=0,
    arrival_idx=7,
    use_edge_feat=True,
    use_mean_agg=True,
    readout_beta=8.0,
    skip_aggregate_save=False,
    seed=42,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    dataset = ArithDataset(data_path, target=target)
    target_mean = dataset.power_mean.to(device)
    target_std = dataset.power_std.to(device)

    node_feature_dim = int(dataset.data[0]["X"].shape[1])
    sample_ea = dataset.data[0].get("edge_attr", None)
    external_edge_attr_dim = int(sample_ea.shape[-1]) if sample_ea is not None else 0
    sample_ns = np.array([dataset.data[i]["X"].shape[0] for i in range(len(dataset))])

    model_kwargs = {
        "node_feature_dim": node_feature_dim,
        "hidden_dim": hidden_dim,
        "num_gnn_layers": num_gnn_layers,
        "dropout": dropout,
        "topo_idx": topo_idx,
        "arrival_idx": arrival_idx,
        "use_edge_feat": use_edge_feat,
        "external_edge_attr_dim": external_edge_attr_dim,
        "use_mean_agg": use_mean_agg,
        "readout_beta": readout_beta,
    }
    probe = DAGTimingGNN(**model_kwargs)
    n_params = sum(p.numel() for p in probe.parameters())
    actual_use_edge_feat = probe.use_edge_feat
    del probe

    print("\n  Starting DAGTimingGNN training")
    print(f"     target={target}, samples={len(dataset)}, X_dim={node_feature_dim}")
    print(f"     model: directed DAG pass by X[:, {topo_idx}] + smooth critical-path readout")
    print(f"     hidden_dim={hidden_dim}, layers={num_gnn_layers}, dropout={dropout}")
    print(f"     edge features: external_dim={external_edge_attr_dim}, "
          f"arrival_feat={'on' if actual_use_edge_feat else 'off'}")
    print(f"     readout_beta={readout_beta}, params={n_params:,}")
    print(f"     loss={w_mse}*Huber + {w_rank}*Rank + {w_list}*ListMLE + {w_scale}*Scale")
    print("     train data: "
          + (f"only N={train_filter_n}" if train_filter_n else "all N")
          + (" + stratified batches" if use_stratified_batch else ""))
    print(f"     eval selection: "
          + (f"fix-N={eval_filter_n} R@10" if eval_filter_n else "mixed-N R@10"))
    print(f"     DataLoader: train_bs={batch_size}, val_bs={val_batch_size}, workers={num_workers}\n")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_taus = []
    fold_test_metrics = []
    best_overall_tau = -1.0
    best_overall_state = None

    for fold_id, (train_idx, val_idx) in enumerate(kf.split(range(len(dataset)))):
        if fold_id < start_fold:
            continue
        if max_folds is not None and fold_id >= max_folds:
            print(f"\n  Reached max_folds={max_folds}, stopping")
            break

        train_idx_full = train_idx
        if train_filter_n is not None:
            train_idx = train_idx[sample_ns[train_idx] == train_filter_n]
            print(f"  train_filter_n={train_filter_n}: {len(train_idx_full)} -> {len(train_idx)}")

        print(f"  {'=' * 60}")
        print(f"  Fold {fold_id}: train={len(train_idx)}, val={len(val_idx)}")
        print(f"  {'=' * 60}")

        rng = np.random.RandomState(1234 + fold_id)
        perm = rng.permutation(len(val_idx))
        half = len(val_idx) // 2
        sel_idx = val_idx[perm[:half]]
        test_idx = val_idx[perm[half:]]
        print(f"     val split: selection={len(sel_idx)}, held-out test={len(test_idx)}")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, sel_idx)
        test_subset = Subset(dataset, test_idx)
        actual_bs = min(batch_size, len(train_idx))
        actual_val_bs = min(val_batch_size, len(sel_idx))

        if use_stratified_batch:
            train_batch_sampler = StratifiedNBatchSampler(
                train_subset, batch_size=actual_bs, drop_last=True,
                shuffle=True, seed=seed + fold_id,
            )
            train_loader = DataLoader(
                train_subset, batch_sampler=train_batch_sampler,
                collate_fn=custom_collate, num_workers=num_workers,
                pin_memory=True, persistent_workers=(num_workers > 0),
            )
        else:
            train_loader = DataLoader(
                train_subset, batch_size=actual_bs, shuffle=True, drop_last=True,
                collate_fn=custom_collate, num_workers=num_workers,
                pin_memory=True, persistent_workers=(num_workers > 0),
            )

        val_loader = DataLoader(
            val_subset, batch_size=actual_val_bs, shuffle=False,
            collate_fn=custom_collate, num_workers=num_workers,
            pin_memory=True, persistent_workers=(num_workers > 0),
        )
        test_loader = DataLoader(
            test_subset, batch_size=min(val_batch_size, max(len(test_idx), 1)),
            shuffle=False, collate_fn=custom_collate, num_workers=num_workers,
            pin_memory=True, persistent_workers=(num_workers > 0),
        )

        model = DAGTimingGNN(**model_kwargs).to(device)
        tau, state, test_metrics = train_one_fold(
            model, train_loader, val_loader,
            device, epochs, lr, weight_decay,
            w_mse, w_rank, w_list, w_scale,
            target_mean, target_std, fold_id,
            eval_filter_n=eval_filter_n,
            w_node=0.0, node_warmup_epochs=0,
            use_multitask=False, w_area=0.0, w_delay=0.0,
            test_loader=test_loader,
        )
        if test_metrics is not None:
            fold_test_metrics.append(test_metrics)

        if state is None:
            print(f"       Fold {fold_id}: no best_state, skip checkpoint")
            continue

        fold_ckpt_path = save_path.replace(".pth", f"_fold{fold_id}.pth")
        _save_ckpt(fold_ckpt_path, state, dataset, model_kwargs, target,
                   fold_id=fold_id, best_tau=tau)
        print(f"       Fold checkpoint saved: {fold_ckpt_path}")

        fold_taus.append((fold_id, tau))
        if tau > best_overall_tau:
            best_overall_tau = tau
            best_overall_state = state

    print(f"\n  {'=' * 60}")
    print("  K-Fold results (validation selection peak):")
    for fold_id, tau in fold_taus:
        print(f"     Fold {fold_id}: tau={tau:+.4f}")
    if fold_taus:
        vals = np.array([tau for _, tau in fold_taus], dtype=float)
        print(f"     mean tau={vals.mean():+.4f} +/- {vals.std():.4f}")
        print(f"     best tau={best_overall_tau:+.4f}")
    print(f"  {'=' * 60}")

    _aggregate_test_metrics(fold_test_metrics, eval_filter_n)

    if skip_aggregate_save:
        print("\n  only_fold mode: skip aggregate save, per-fold checkpoints are saved")
    elif best_overall_state is not None:
        _save_ckpt(save_path, best_overall_state, dataset, model_kwargs, target,
                   best_tau=best_overall_tau)
        print(f"\n  Best model saved to: {save_path}")


def main():
    # 行缓冲：重定向到文件时也实时刷新日志，避免 epoch 进度被块缓冲“卡住”的假象
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description="DAG-GNN delay proxy trainer")
    parser.add_argument("--mode", choices=["B", "C", "default"], default="B",
                        help="B=stratified batch all data, C=only train N=eval_filter_n, default=plain all data")
    parser.add_argument("--data", default="dataset/glitch_power_data_16bit_v2_13k_edge10.pt")
    parser.add_argument("--save_suffix", default="")
    parser.add_argument("--folds", type=int, default=1)
    parser.add_argument("--start_fold", type=int, default=0)
    parser.add_argument("--only_fold", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=96)
    parser.add_argument("--num_gnn_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_filter_n", type=int, default=730)
    parser.add_argument("--target", choices=["delay", "area", "power"], default="delay")
    parser.add_argument("--w_mse", type=float, default=0.7)
    parser.add_argument("--w_rank", type=float, default=0.2)
    parser.add_argument("--w_list", type=float, default=0.3)
    parser.add_argument("--w_scale", type=float, default=0.3)
    parser.add_argument("--topo_idx", type=int, default=0)
    parser.add_argument("--arrival_idx", type=int, default=7)
    parser.add_argument("--no_edge_feat", action="store_true")
    parser.add_argument("--sum_agg", action="store_true",
                        help="Use sum aggregation instead of degree-normalized mean")
    parser.add_argument("--readout_beta", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default=None,
                        help="权重输出目录; 不填则自动用日志(stdout重定向)所在目录, 否则 dataset/")
    args = parser.parse_args()

    target_tag = "" if args.target == "delay" else f"_{args.target}"
    suffix = args.save_suffix or ""
    if args.mode == "B":
        save_path = f"dataset/glitch_power_dag_gnn_delay_B{target_tag}{suffix}.pth"
        train_filter_n = None
        use_stratified_batch = True
    elif args.mode == "C":
        save_path = f"dataset/glitch_power_dag_gnn_delay_C{target_tag}{suffix}.pth"
        train_filter_n = args.eval_filter_n if args.eval_filter_n > 0 else None
        use_stratified_batch = False
    else:
        save_path = f"dataset/glitch_power_dag_gnn_delay{target_tag}{suffix}.pth"
        train_filter_n = None
        use_stratified_batch = False

    out_dir = resolve_out_dir(args.out_dir)
    save_path = place(save_path, out_dir)
    print(f"  💾 权重输出目录: {out_dir}")

    if args.only_fold is not None:
        start_fold = args.only_fold
        max_folds = args.only_fold + 1
        skip_aggregate_save = True
    else:
        start_fold = args.start_fold
        max_folds = args.folds
        skip_aggregate_save = False

    train_dag_gnn_delay(
        data_path=args.data,
        save_path=save_path,
        max_folds=max_folds,
        start_fold=start_fold,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_gnn_layers,
        dropout=args.dropout,
        w_mse=args.w_mse,
        w_rank=args.w_rank,
        w_list=args.w_list,
        w_scale=args.w_scale,
        num_workers=args.num_workers,
        eval_filter_n=None if args.eval_filter_n <= 0 else args.eval_filter_n,
        train_filter_n=train_filter_n,
        use_stratified_batch=use_stratified_batch,
        target=args.target,
        topo_idx=args.topo_idx,
        arrival_idx=args.arrival_idx,
        use_edge_feat=not args.no_edge_feat,
        use_mean_agg=not args.sum_agg,
        readout_beta=args.readout_beta,
        skip_aggregate_save=skip_aggregate_save,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
