import argparse
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proxy_mlp import OneHotGIN
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
    print("  🧪 TEST 无偏泛化 (主指标, 对外汇报用):")
    print(f"     per-N 加权 : τ = {st_tau[0]:+.4f} ± {st_tau[1]:.4f}"
          f" | ρ = {st_rho[0]:+.4f} ± {st_rho[1]:.4f}"
          f" | R@10 = {st_r10[0]:.4f} ± {st_r10[1]:.4f}")
    if eval_filter_n is not None:
        fx_tau = agg(lambda m: m["fix"]["tau"] if m["fix"] else None)
        fx_r10 = agg(lambda m: m["fix"]["recalls"][0.10] if m["fix"] else None)
        print(f"     fix-N={eval_filter_n}  : τ = {fx_tau[0]:+.4f} ± {fx_tau[1]:.4f}"
              f" | R@10 = {fx_r10[0]:.4f} ± {fx_r10[1]:.4f}")


def train_onehot_gin(
    data_path="dataset/glitch_power_data_16bit_v2.pt",
    save_path="dataset/glitch_power_onehot_gin.pth",
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
    dropout=0.0,
    w_mse=0.5,
    w_rank=0.2,
    w_list=0.5,
    w_scale=0.5,
    num_workers=4,
    eval_filter_n=730,
    train_filter_n=None,
    use_stratified_batch=False,
    target="power",
    onehot_start=3,
    onehot_dim=4,
    skip_aggregate_save=False,
):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  🖥  Device: {device}")

    if target == "area" and eval_filter_n is not None:
        print("  ℹ  target=area: 关闭 fixed-N 评估/选模，使用 mixed-N R@10%")
        eval_filter_n = None

    dataset = ArithDataset(data_path, target=target)
    power_mean = dataset.power_mean.to(device)
    power_std = dataset.power_std.to(device)
    actual_dim = dataset.data[0]["X"].shape[1]
    if actual_dim < onehot_start + onehot_dim:
        raise ValueError(
            f"X dim={actual_dim} 不足以读取 onehot slice "
            f"[{onehot_start}:{onehot_start + onehot_dim}]"
        )

    sample_ns = np.array([dataset.data[i]["X"].shape[0] for i in range(len(dataset))])
    model_probe = OneHotGIN(
        node_feature_dim=actual_dim,
        hidden_dim=hidden_dim,
        num_gnn_layers=num_gnn_layers,
        dropout=dropout,
        onehot_start=onehot_start,
        onehot_dim=onehot_dim,
    )
    n_params = sum(p.numel() for p in model_probe.parameters())
    del model_probe

    print(f"\n  🚀 开始 OneHotGIN {target} 预测")
    print(f"     模型: pure GIN, 单向 sum aggregation, graph-level multi-layer sum readout")
    print(f"     输入特征: 仅 X[:, {onehot_start}:{onehot_start + onehot_dim}] 节点类型 one-hot")
    print(f"     忽略: stage/col/idx/arrival/edge_attr/physics/node_power/area/delay aux")
    print(f"     参数量: {n_params:,}")
    print(f"     hidden_dim={hidden_dim}, layers={num_gnn_layers}, dropout={dropout}")
    print(f"     Loss: {w_mse}×Huber + {w_rank}×RankLoss + {w_list}×ListMLE + {w_scale}×ScaleLoss")
    print("     训练数据: "
          + (f"只用 N={train_filter_n} 子集" if train_filter_n else "全集")
          + (" + stratified batch (同 N 同 batch)" if use_stratified_batch else ""))
    print(f"     评估: best 选择基于 "
          + (f"fix-N={eval_filter_n} R@10%" if eval_filter_n else "mixed-N R@10%"))
    print(f"     DataLoader: train_bs={batch_size}, val_bs={val_batch_size}, num_workers={num_workers}\n")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_taus = []
    fold_test_metrics = []
    best_overall_tau = -1.0
    best_overall_state = None

    for fold_id, (train_idx, val_idx) in enumerate(kf.split(range(len(dataset)))):
        if fold_id < start_fold:
            continue
        if max_folds is not None and fold_id >= max_folds:
            print(f"\n  ⏭  达到 max_folds={max_folds}，提前结束")
            break

        train_idx_full = train_idx
        if train_filter_n is not None:
            train_idx = train_idx[sample_ns[train_idx] == train_filter_n]
            print(f"  📌 train_filter_n={train_filter_n}: 训练样本 {len(train_idx_full)} → {len(train_idx)}")

        print(f"  {'=' * 60}")
        print(f"  Fold {fold_id}: train={len(train_idx)}, val={len(val_idx)}")
        print(f"  {'=' * 60}")

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
            train_batch_sampler = StratifiedNBatchSampler(
                train_subset, batch_size=actual_bs, drop_last=True,
                shuffle=True, seed=42 + fold_id,
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

        model = OneHotGIN(
            node_feature_dim=actual_dim,
            hidden_dim=hidden_dim,
            num_gnn_layers=num_gnn_layers,
            dropout=dropout,
            onehot_start=onehot_start,
            onehot_dim=onehot_dim,
        ).to(device)

        tau, state, test_metrics = train_one_fold(
            model, train_loader, val_loader,
            device, epochs, lr, weight_decay,
            w_mse, w_rank, w_list, w_scale,
            power_mean, power_std, fold_id,
            eval_filter_n=eval_filter_n,
            w_node=0.0, node_warmup_epochs=0,
            use_multitask=False, w_area=0.0, w_delay=0.0,
            test_loader=test_loader,
        )
        if test_metrics is not None:
            fold_test_metrics.append(test_metrics)

        if state is None:
            print(f"       ⚠️ Fold {fold_id} 没有产生 best_state，跳过保存")
            continue

        fold_ckpt_path = save_path.replace(".pth", f"_fold{fold_id}.pth")
        torch.save({
            "model_state_dict": state,
            "model_class": "OneHotGIN",
            "target": target,
            "node_feature_dim": actual_dim,
            "hidden_dim": hidden_dim,
            "num_gnn_layers": num_gnn_layers,
            "dropout": dropout,
            "onehot_start": onehot_start,
            "onehot_dim": onehot_dim,
            "use_onehot_only": True,
            "use_mean_agg": False,
            "use_edge_feat": False,
            "external_edge_attr_dim": 0,
            "use_typed_edges": False,
            "use_multitask": False,
            "use_jk_pool": False,
            "use_gin": True,
            "use_pure_gin": False,
            "power_mean": dataset.power_mean.item(),
            "power_std": dataset.power_std.item(),
            "fold_id": fold_id,
            "best_tau": tau,
        }, fold_ckpt_path)
        print(f"       💾 Fold {fold_id} ckpt 已保存: {fold_ckpt_path}")

        fold_taus.append((fold_id, tau))
        if tau > best_overall_tau:
            best_overall_tau = tau
            best_overall_state = state

    print(f"\n  {'=' * 60}")
    print("  📊 K-Fold 结果 (val 选择集峰值, 乐观上界):")
    for fold_id, tau in fold_taus:
        print(f"     Fold {fold_id}: τ = {tau:+.4f}")
    if fold_taus:
        vals = np.array([t for _, t in fold_taus], dtype=float)
        print(f"     平均 τ = {vals.mean():+.4f} ± {vals.std():.4f}")
        print(f"     最佳 τ = {best_overall_tau:+.4f}")
    print(f"  {'=' * 60}")

    _aggregate_test_metrics(fold_test_metrics, eval_filter_n)

    if skip_aggregate_save:
        print("\n  ℹ  --only_fold 模式: 跳过聚合 save, per-fold ckpt 已保存")
    elif best_overall_state is not None:
        torch.save({
            "model_state_dict": best_overall_state,
            "model_class": "OneHotGIN",
            "target": target,
            "node_feature_dim": actual_dim,
            "hidden_dim": hidden_dim,
            "num_gnn_layers": num_gnn_layers,
            "dropout": dropout,
            "onehot_start": onehot_start,
            "onehot_dim": onehot_dim,
            "use_onehot_only": True,
            "use_mean_agg": False,
            "use_edge_feat": False,
            "external_edge_attr_dim": 0,
            "use_typed_edges": False,
            "use_multitask": False,
            "use_jk_pool": False,
            "use_gin": True,
            "use_pure_gin": False,
            "power_mean": dataset.power_mean.item(),
            "power_std": dataset.power_std.item(),
        }, save_path)
        print(f"\n  ✅ 最佳模型已保存至: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Pure one-hot GIN target proxy")
    parser.add_argument("--mode", choices=["B", "C", "default"], default="B",
                        help="B=stratified batch 全集, C=只训练 N=730, default=普通全集")
    parser.add_argument("--data", default="dataset/glitch_power_data_16bit_v2.pt")
    parser.add_argument("--save_suffix", default="")
    parser.add_argument("--folds", type=int, default=1,
                        help="max_folds: 5 表示跑完整 5-fold")
    parser.add_argument("--start_fold", type=int, default=0)
    parser.add_argument("--only_fold", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=96)
    parser.add_argument("--num_gnn_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_filter_n", type=int, default=730,
                        help="fixed-N 评估子集; <=0 表示关闭。target=area 时会自动关闭")
    parser.add_argument("--target", choices=["power", "area", "delay"], default="power")
    parser.add_argument("--w_mse", type=float, default=0.5)
    parser.add_argument("--w_rank", type=float, default=0.2)
    parser.add_argument("--w_list", type=float, default=0.5)
    parser.add_argument("--w_scale", type=float, default=0.5)
    parser.add_argument("--onehot_start", type=int, default=3)
    parser.add_argument("--onehot_dim", type=int, default=4)
    args = parser.parse_args()

    suffix = args.save_suffix or ""
    tgt_tag = "" if args.target == "power" else f"_{args.target}"
    if args.mode == "B":
        save_path = f"dataset/glitch_power_onehot_gin_B{tgt_tag}{suffix}.pth"
        train_filter_n = None
        use_stratified_batch = True
    elif args.mode == "C":
        save_path = f"dataset/glitch_power_onehot_gin_C{tgt_tag}{suffix}.pth"
        train_filter_n = 730
        use_stratified_batch = False
    else:
        save_path = f"dataset/glitch_power_onehot_gin{tgt_tag}{suffix}.pth"
        train_filter_n = None
        use_stratified_batch = False

    if args.only_fold is not None:
        start_fold = args.only_fold
        max_folds = args.only_fold + 1
        skip_aggregate_save = True
    else:
        start_fold = args.start_fold
        max_folds = args.folds
        skip_aggregate_save = False

    eval_filter_n = None if args.eval_filter_n <= 0 else args.eval_filter_n

    train_onehot_gin(
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
        eval_filter_n=eval_filter_n,
        train_filter_n=train_filter_n,
        use_stratified_batch=use_stratified_batch,
        target=args.target,
        onehot_start=args.onehot_start,
        onehot_dim=args.onehot_dim,
        skip_aggregate_save=skip_aggregate_save,
    )


if __name__ == "__main__":
    main()
