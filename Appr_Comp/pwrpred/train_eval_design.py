#!/usr/bin/env python
"""设计级 XA 功耗预测可行性：121 设计 / 13 run。
S1 = 纯结构特征（RTL 即可）；S2 = 结构 + DC 副产品（area_dc, delay）。
考核：leave-one-run-out 的全局与 run 内 Spearman / MAPE。"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

BASE = "/tmp/claude-1000/-home-lee-Power-das/c973f85e-6eb5-4da8-b295-c00dfc39f184/scratchpad/pwrpred"

S1 = ["n_fa", "n_ha", "n_ct42e", "n_c32", "n_c22", "n_c42", "n_cells",
      "sum_dyn_lib", "sum_area_lib", "n_pp_active", "n_const0", "n_const1",
      "n_wire", "n_assign", "booth"]
S2 = S1 + ["area_dc", "delay"]

def loro_eval(df, cols, label, model="gbdt"):
    df = df.dropna(subset=cols + ["power_xa_mw"]).reset_index(drop=True)
    y = df["power_xa_mw"].values
    preds = np.full(len(y), np.nan)
    for run in df["run"].unique():
        te = (df["run"] == run).values; tr = ~te
        if model == "gbdt":
            mdl = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08,
                                                min_samples_leaf=4, random_state=0)
        else:
            mdl = Ridge(1.0)
        X = df[cols].values.astype(float)
        mdl.fit(X[tr], y[tr]); preds[te] = mdl.predict(X[te])
    pear = pearsonr(y, preds)[0]; spear = spearmanr(y, preds)[0]
    mape = np.mean(np.abs(preds - y) / y) * 100
    # run 内排序
    per_run = []
    for run, g in df.assign(pred=preds).groupby("run"):
        if len(g) >= 5 and g["power_xa_mw"].nunique() > 2:
            per_run.append(spearmanr(g["power_xa_mw"], g["pred"])[0])
    print(f"{label:<42s} n={len(y):<4d} pearson={pear:.4f} spearman={spear:.4f} "
          f"MAPE={mape:5.2f}%  run内spearman: mean={np.mean(per_run):.3f} min={np.min(per_run):.3f} (n_runs={len(per_run)})")

def main():
    df = pd.read_csv(f"{BASE}/design_xa.csv")
    print(f"designs={len(df)} runs={df.run.nunique()}  XA power range: "
          f"{df.power_xa_mw.min():.3f}-{df.power_xa_mw.max():.3f} mW")
    # 裸基线（不训模型）
    for c in ["area_dc", "sum_dyn_lib", "n_cells", "n_pp"]:
        d = df.dropna(subset=[c])
        glob_sp = spearmanr(d.power_xa_mw, d[c])[0]
        pr = [spearmanr(g.power_xa_mw, g[c])[0] for _, g in d.groupby("run") if len(g) >= 5]
        print(f"baseline raw {c:<14s} global_spearman={glob_sp:.4f}  run内 mean={np.mean(pr):.3f} min={np.min(pr):.3f}")
    print()
    loro_eval(df, S1, "S1 structure-only, GBDT (LORO)")
    loro_eval(df, S1, "S1 structure-only, Ridge (LORO)", model="ridge")
    loro_eval(df, S2, "S2 +DC area/delay, GBDT (LORO)")
    loro_eval(df, S2, "S2 +DC area/delay, Ridge (LORO)", model="ridge")

if __name__ == "__main__":
    main()
