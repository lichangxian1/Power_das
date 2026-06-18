#!/usr/bin/env python3
"""GBDT sanity baseline for area prediction on handcrafted graph aggregates.

目的: 量化 (1) GNN 是否输给简单模型 (2) 同-N 内 area 信号用聚合特征能学到多少。
特征全部是 routing/结构聚合量 — 不喂 N 以外的捷径之外的东西是不可能的
(N 本身就是强特征), 所以重点看 per-N / fix-N=730 的 τ。
"""
import argparse
import collections

import numpy as np
import torch
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

MAX_STAGE = 10
MAX_COL = 32


def graph_features(item):
    X = item["X"].numpy()
    ei = item["edge_index"].numpy()
    ea = item["edge_attr"].numpy()
    N = X.shape[0]
    stage = X[:, 0].astype(int)
    col = X[:, 1].astype(int)
    typ = X[:, 3:7].argmax(1)

    f = [float(N)]
    # type counts
    f += [float((typ == t).sum()) for t in range(4)]
    # per-(stage,type) counts
    st = np.zeros((MAX_STAGE, 4))
    np.add.at(st, (np.clip(stage, 0, MAX_STAGE - 1), typ), 1.0)
    f += st.flatten().tolist()
    # per-(col,type) counts
    ct = np.zeros((MAX_COL, 4))
    np.add.at(ct, (np.clip(col, 0, MAX_COL - 1), typ), 1.0)
    f += ct.flatten().tolist()
    # edge_attr aggregates (sum + mean per dim)
    if ea.shape[0] > 0:
        f += ea.sum(0).tolist() + ea.mean(0).tolist()
    else:
        f += [0.0] * (2 * ea.shape[1])
    # out-degree stats (fanout drives buffering/sizing)
    deg = np.bincount(ei[0], minlength=N).astype(float)
    f += [deg.max(), deg.mean(), deg.std(),
          float((deg >= 3).sum()), float((deg >= 5).sum())]
    # carry edges per dst col (carry chains concentrate sizing pressure)
    is_carry = ea[:, 1] > 0.5 if ea.shape[0] > 0 else np.zeros(0, bool)
    cc = np.zeros(MAX_COL)
    if is_carry.any():
        np.add.at(cc, np.clip(col[ei[1][is_carry]], 0, MAX_COL - 1), 1.0)
    f += cc.tolist()
    return f


def tau_of(pred, true):
    t, _ = stats.kendalltau(pred, true)
    return 0.0 if np.isnan(t) else t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/glitch_power_data_16bit_v2_11k_edge10.pt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = torch.load(args.data, map_location="cpu", weights_only=False)
    feats = np.array([graph_features(d) for d in data], dtype=np.float32)
    areas = np.array([float(d["area"]) for d in data])
    Ns = np.array([d["X"].shape[0] for d in data])
    print(f"samples={len(data)} feat_dim={feats.shape[1]}")

    Xtr, Xte, ytr, yte, ntr, nte = train_test_split(
        feats, areas, Ns, test_size=0.2, random_state=args.seed)

    model = HistGradientBoostingRegressor(
        max_iter=600, learning_rate=0.05, max_depth=None,
        early_stopping=True, validation_fraction=0.1, random_state=args.seed)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    print(f"\n[mix]  n={len(yte)}  tau={tau_of(pred, yte):+.4f}  "
          f"MAPE={np.mean(np.abs(pred - yte) / yte) * 100:.2f}%")

    # per-N weighted tau
    taus, ws = [], []
    for Nv, cnt in collections.Counter(nte).items():
        if cnt < 30:
            continue
        m = nte == Nv
        taus.append(tau_of(pred[m], yte[m]))
        ws.append(m.sum())
    if taus:
        w = np.array(ws) / sum(ws)
        print(f"[perN] buckets={len(taus)}  weighted_tau={np.dot(w, taus):+.4f}  "
              f"simple_tau={np.mean(taus):+.4f}")
    m = nte == 730
    if m.sum() > 30:
        print(f"[fix N=730] n={m.sum()}  tau={tau_of(pred[m], yte[m]):+.4f}")

    # 对照: 只用 N 一个特征 → 捷径基线
    nm = HistGradientBoostingRegressor(max_iter=200, random_state=args.seed)
    nm.fit(ntr.reshape(-1, 1).astype(float), ytr)
    np_pred = nm.predict(nte.reshape(-1, 1).astype(float))
    print(f"\n[N-only baseline] mix tau={tau_of(np_pred, yte):+.4f} "
          f"(per-N tau == 0 by construction)")


if __name__ == "__main__":
    main()
