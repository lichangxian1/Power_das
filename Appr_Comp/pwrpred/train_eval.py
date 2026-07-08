#!/usr/bin/env python
"""可行性实验：用 TT 特征 / yosys 网表特征预测 DC (compile_medium) 的 dyn_mw / area / tmax。"""
import re, sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

BASE = "/tmp/claude-1000/-home-lee-Power-das/c973f85e-6eb5-4da8-b295-c00dfc39f184/scratchpad/pwrpred"

def parse_yosys_stat(path):
    rows, cur = [], None
    gate_types = set()
    for line in open(path):
        m = re.match(r"^=== (\S+) ===", line)
        if m:
            if cur: rows.append(cur)
            cur = {"name": m.group(1)}
            continue
        if cur is None: continue
        m = re.match(r"^\s+Number of cells:\s+(\d+)", line)
        if m: cur["yos_ncells"] = int(m.group(1)); continue
        m = re.match(r"^\s+Number of wires:\s+(\d+)", line)
        if m: cur["yos_nwires"] = int(m.group(1)); continue
        m = re.match(r"^\s+\$_(\w+)_\s+(\d+)", line)
        if m:
            g = "yos_" + m.group(1)
            cur[g] = int(m.group(2)); gate_types.add(g)
    if cur: rows.append(cur)
    df = pd.DataFrame(rows).fillna(0)
    return df, sorted(gate_types)

def topk_overlap(y_true, y_pred, k):
    a = set(np.argsort(y_true)[:k]); b = set(np.argsort(y_pred)[:k])
    return len(a & b) / k

def pairwise_acc(y_true, y_pred, n_pairs=20000, seed=0):
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(y_true), n_pairs); j = rng.integers(0, len(y_true), n_pairs)
    m = y_true[i] != y_true[j]
    return float(np.mean(np.sign(y_true[i]-y_true[j])[m] == np.sign(y_pred[i]-y_pred[j])[m]))

def evaluate(df, feat_cols, target, split, label, model="gbdt"):
    df = df[np.isfinite(df[target].values.astype(float)) & (df[target].values.astype(float) != 0)].reset_index(drop=True)
    X = df[feat_cols].values.astype(float); y = df[target].values.astype(float)
    preds = np.full(len(y), np.nan)
    if split == "cv":
        for tr, te in KFold(5, shuffle=True, random_state=42).split(X):
            mdl = make_model(model); mdl.fit(X[tr], y[tr]); preds[te] = mdl.predict(X[te])
        yt, yp = y, preds
    else:  # family holdout: train rand*, test structured
        tr = df["family"].str.startswith("rand").values
        te = ~tr
        mdl = make_model(model); mdl.fit(X[tr], y[tr]); yp = mdl.predict(X[te]); yt = y[te]
    pear = pearsonr(yt, yp)[0]; spear = spearmanr(yt, yp)[0]
    mape = float(np.mean(np.abs(yp - yt) / np.abs(yt))) * 100
    k = max(10, len(yt) // 20)
    tk = topk_overlap(yt, yp, k)
    pw = pairwise_acc(yt, yp)
    print(f"{label:<38s} {target:<7s} n_test={len(yt):<5d} pearson={pear:.4f} spearman={spear:.4f} MAPE={mape:5.2f}% top{k}={tk:.2f} pairAcc={pw:.4f}")
    return spear

def make_model(kind):
    if kind == "gbdt":
        return HistGradientBoostingRegressor(max_iter=500, learning_rate=0.08, max_depth=None,
                                             l2_regularization=1e-3, random_state=0)
    return Ridge(alpha=1.0)

def main():
    tt = pd.read_csv(f"{BASE}/tt42.csv")
    yos, gate_cols = parse_yosys_stat(f"{BASE}/yosys_stat42.txt")
    df = tt.merge(yos, on="name", how="inner")
    print(f"merged: {len(df)} cells; gate types: {gate_cols}")

    tt_analytic = [c for c in tt.columns if re.search(r"_(p1|toggle|infl_\d|infl_tot|ones)$", c)] + ["v_err_l1", "hamming_v"]
    tt_bits = [c for c in tt.columns if re.search(r"_bit\d+$", c)]
    yos_feats = ["yos_ncells", "yos_nwires"] + gate_cols

    sets = {
        "TT-analytic only (zero cost)": tt_analytic,
        "TT-analytic + raw bits": tt_analytic + tt_bits,
        "yosys netlist only": yos_feats,
        "TT + yosys netlist": tt_analytic + tt_bits + yos_feats,
    }
    for target in ["dyn_mw", "area", "tmax"]:
        print(f"\n### target = {target} (random 5-fold CV)")
        # trivial baselines
        for bl_col, bl_name in [("yos_ncells", "baseline: yosys gate count"),
                                ("sum_toggle", "baseline: sum toggle")]:
            sp = spearmanr(df[target], df[bl_col])[0]
            print(f"{bl_name:<38s} {target:<7s} spearman={sp:.4f} (raw, no model)")
        for name, cols in sets.items():
            evaluate(df, cols, target, "cv", name)
    print(f"\n### family holdout (train rand/rand_zb n={df['family'].str.startswith('rand').sum()}, test structured n={(~df['family'].str.startswith('rand')).sum()})")
    for target in ["dyn_mw", "area"]:
        for name, cols in sets.items():
            evaluate(df, cols, target, "family", name)

if __name__ == "__main__":
    main()
