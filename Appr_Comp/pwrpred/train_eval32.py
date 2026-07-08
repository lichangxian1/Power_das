#!/usr/bin/env python
"""comp32/comp22 库复验：随机 CV + 跨类型（train 3:2 -> test 2:2）OOD 检验。"""
import re
import numpy as np
import pandas as pd
from train_eval import parse_yosys_stat, evaluate, BASE, make_model, topk_overlap, pairwise_acc
from scipy.stats import spearmanr, pearsonr

def main():
    tt = pd.read_csv(f"{BASE}/tt32.csv")
    y32, g32 = parse_yosys_stat(f"{BASE}/yosys_stat_comp32.txt")
    y22, g22 = parse_yosys_stat(f"{BASE}/yosys_stat_comp22.txt")
    yos = pd.concat([y32, y22], ignore_index=True).fillna(0)
    gate_cols = sorted(set(g32) | set(g22))
    for c in gate_cols:
        if c not in yos.columns: yos[c] = 0
    df = tt.merge(yos, on="name", how="inner")
    df["family"] = df["ctype"].map({32: "rand_32", 22: "s_22"}).fillna("rand_32")  # 复用 evaluate 的 family holdout: train 32 test 22
    print(f"merged: {len(df)} (32: {(df.ctype==32).sum()}, 22: {(df.ctype==22).sum()})")

    tt_analytic = [c for c in tt.columns if re.search(r"_(p1|toggle|infl_\d|infl_tot|ones)$", c)] + ["v_err_l1", "hamming_v", "n_in"]
    tt_bits = [c for c in tt.columns if re.search(r"_bit\d+$", c)]
    yos_feats = ["yos_ncells", "yos_nwires"] + gate_cols
    sets = {
        "TT-analytic only (zero cost)": tt_analytic,
        "TT-analytic + raw bits": tt_analytic + tt_bits,
        "yosys netlist only": yos_feats,
        "TT + yosys netlist": tt_analytic + tt_bits + yos_feats,
    }
    d32 = df[df.ctype == 32].reset_index(drop=True)
    for target in ["dyn_w", "area", "tmax"]:
        print(f"\n### comp32 only (n={len(d32)}) target={target} random 5-fold CV")
        print(f"{'baseline: yosys gate count':<38s} spearman={spearmanr(d32[target], d32['yos_ncells'])[0]:.4f}")
        for name, cols in sets.items():
            evaluate(d32, cols, target, "cv", name)
    print("\n### OOD: train comp32 -> test comp22 (dyn_w)")
    for name, cols in sets.items():
        evaluate(df, cols, "dyn_w", "family", name)

if __name__ == "__main__":
    main()
